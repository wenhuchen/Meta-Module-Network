from Attention.net_utils import FC, MLP, LayerNorm
import torch.optim as optim
from torch import nn
import torch
from torch import autograd
import torch.nn.functional as F
import Constants
import math
import numpy as np
from enum import Enum
from API_provider import APIS
device = torch.device('cuda')


class Type(Enum):
    FUNC = 1
    COMMA = 2
    LEFT = 3
    RIGHT = 4
    SEP = 5
    VAR = 6


class Beam(object):
    ''' Beam search '''

    def __init__(self, size, de_vocab, device=False):

        self.size = size
        self._done = False

        # The score for each translation on the beam.
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.full((size,), Constants.PAD, dtype=torch.long, device=device)]
        self.next_ys[0][0] = Constants.SOS
        self.finished = [False for _ in range(size)]
        self.de_vocab = de_vocab
        self.de_ivocab = {v: k for k, v in de_vocab.items()}

        self.FUNCS = [de_vocab[k] for k in APIS if k in de_vocab]

        self.COMMA, self.LEFT, self.RIGHT, self.SEP = [de_vocab[',']], [de_vocab['(']], [de_vocab[')']], [de_vocab[';']]
        SPECIAL = set(self.FUNCS + self.COMMA + self.LEFT + self.RIGHT + self.SEP +
                      [Constants.SOS, Constants.EOS, Constants.PAD, Constants.UNK])
        self.VARS = list(set(range(len(de_vocab))) - SPECIAL)

        self.imapping = {}
        self.imapping.update({v: Type.FUNC for v in self.FUNCS})
        self.imapping.update({v: Type.VAR for v in self.VARS})
        self.imapping.update({self.COMMA[0]: Type.COMMA, self.LEFT[0]: Type.LEFT,
                              self.RIGHT[0]: Type.RIGHT, self.SEP[0]: Type.SEP})
        self.placeholder = -1000

        self.next_mapping = {}
        self.next_mapping[Type.FUNC] = torch.FloatTensor(len(de_vocab)).fill_(self.placeholder).to(device)
        self.next_mapping[Type.FUNC][self.LEFT[0]] = 0

        self.next_mapping[Type.COMMA] = torch.FloatTensor(len(de_vocab)).fill_(self.placeholder).to(device)
        self.next_mapping[Type.COMMA][self.VARS] = 0

        self.next_mapping[Type.LEFT] = torch.FloatTensor(len(de_vocab)).fill_(self.placeholder).to(device)
        self.next_mapping[Type.LEFT][self.RIGHT[0]] = 0
        self.next_mapping[Type.LEFT][self.VARS] = 0

        self.next_mapping[Type.RIGHT] = torch.FloatTensor(len(de_vocab)).fill_(self.placeholder).to(device)
        self.next_mapping[Type.RIGHT][self.SEP[0]] = 0
        self.next_mapping[Type.RIGHT][Constants.EOS] = 0

        self.next_mapping[Type.SEP] = torch.FloatTensor(len(de_vocab)).fill_(self.placeholder).to(device)
        self.next_mapping[Type.SEP][self.FUNCS] = 0

        self.next_mapping[Type.VAR] = torch.FloatTensor(len(de_vocab)).fill_(self.placeholder).to(device)
        self.next_mapping[Type.VAR][self.COMMA[0]] = 0
        self.next_mapping[Type.VAR][self.RIGHT[0]] = 0

        self.next_mask = [self.next_mapping[Type.SEP] for _ in range(self.size)]

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def idx2text(self, string):
        out = []
        for idx, s in enumerate(string):
            if s == Constants.EOS:
                break
            else:
                out.append(self.de_ivocab[s])
        return out

    def advance(self, word_prob):
        "Update beam status and check if finished or not."
        num_words = word_prob.size(1)

        for i in range(self.size):
            if self.finished[i]:
                word_prob[i, :].fill_(self.placeholder)
                word_prob[i, Constants.PAD].fill_(0)
            else:
                word_prob[i, :] += self.next_mask[i]

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
        else:
            beam_lk = word_prob[0]

        flat_beam_lk = beam_lk.view(-1)

        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)

        self.next_ys.append(best_scores_id - prev_k * num_words)
        for i in range(self.size):
            word = self.next_ys[-1][i].item()
            if word in [Constants.EOS, Constants.PAD, Constants.UNK]:
                self.next_mask[i] = None
            else:
                self.next_mask[i] = self.next_mapping[self.imapping[word]]

        self.finished = []
        for i in range(self.size):
            self.finished.append(self.next_ys[-1][i].item() in [Constants.EOS, Constants.PAD])

        if all(self.finished):
            self._done = True
        #self._done = self.finished[0]

        return self._done

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep."

        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[Constants.SOS] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k, tensor=True):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]

        idxs = list(map(lambda x: x.item(), hyp[::-1]))

        if tensor:
            return idxs
        else:
            return self.idx2text(idxs)


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class AttFlat(nn.Module):
    def __init__(self, hidden_size, flat_out_size, dropout):
        super(AttFlat, self).__init__()

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=hidden_size,
            out_size=1,
            dropout_r=dropout,
            use_relu=True
        )

        self.linear_merge = nn.Linear(hidden_size, flat_out_size)

    def forward(self, x):
        att = self.mlp(x)
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(1):
            att_list.append(torch.sum(att[:, :, i: i + 1] * x, dim=1))

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)
        return x_atted


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1).type(torch.bool)
    return padding_mask


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(torch.ones((len_s, len_s), device=seq.device), diagonal=1).type(torch.bool)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn

    def step_forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_query = dec_input[:, -1, :].unsqueeze(1)
        slf_attn_mask = slf_attn_mask[:, -1, :].unsqueeze(1)
        dec_enc_attn_mask = dec_enc_attn_mask[:, -1, :].unsqueeze(1)
        non_pad_mask = non_pad_mask[:, -1, :].unsqueeze(1)

        dec_output, dec_slf_attn = self.slf_attn(
            dec_query, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class TransformerDecoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, vocab_size, d_word_vec, n_layers, d_model, n_head, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        d_k = d_model // n_head
        d_v = d_model // n_head
        d_inner = d_model * 4

        self.word_emb = nn.Embedding(vocab_size, 300, padding_idx=Constants.PAD)
        self.src_proj = nn.Linear(300, d_word_vec)

        self.post_word_emb = PositionalEmbedding(d_model=d_word_vec)

        self.enc_layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.tgt_word_prj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tgt_seq, src_seq):
        # -- Encode source
        non_pad_mask = get_non_pad_mask(src_seq)
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)

        enc_inp = self.src_proj(self.word_emb(src_seq)) + self.post_word_emb(src_seq)

        for enc_layer in self.enc_layer_stack:
            enc_inp, _ = enc_layer(enc_inp, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)

        enc_output = enc_inp

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)
        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.src_proj(self.word_emb(tgt_seq)) + self.post_word_emb(tgt_seq)
        #dec_output += vis_feat.unsqueeze(1)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(dec_output, enc_output,
                                                               non_pad_mask=non_pad_mask,
                                                               slf_attn_mask=slf_attn_mask,
                                                               dec_enc_attn_mask=dec_enc_attn_mask)

        logits = self.tgt_word_prj(dec_output)
        return logits

    def translate_batch(self, de_vocab, src_seq, max_token_seq_len=30):
        with torch.no_grad():
            # -- Encode source
            non_pad_mask = get_non_pad_mask(src_seq)
            slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)

            enc_inp = self.src_proj(self.word_emb(src_seq)) + self.post_word_emb(src_seq)

            for layer in self.enc_layer_stack:
                enc_inp, _ = layer(enc_inp, non_pad_mask, slf_attn_mask)
            enc_output = enc_inp

            trg_seq = torch.full((src_seq.size(0), 1), Constants.SOS, dtype=torch.long, device=src_seq.device)

            dec_outputs = []
            for len_dec_seq in range(0, max_token_seq_len + 1):
                # -- Prepare masks
                non_pad_mask = get_non_pad_mask(trg_seq)
                slf_attn_mask_subseq = get_subsequent_mask(trg_seq)
                slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=trg_seq, seq_q=trg_seq)
                slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
                dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=trg_seq)

                dec_output = self.src_proj(self.word_emb(trg_seq)) + self.post_word_emb(trg_seq)
                #dec_output += vis_feat.unsqueeze(1)

                if len_dec_seq == 0:
                    dec_outputs.append(dec_output)
                else:
                    dec_outputs[0] = dec_output

                for i, dec_layer in enumerate(self.layer_stack):
                    tmp = dec_layer.step_forward(
                        dec_outputs[i], enc_output,
                        non_pad_mask=non_pad_mask,
                        slf_attn_mask=slf_attn_mask,
                        dec_enc_attn_mask=dec_enc_attn_mask)

                    if i == len(self.layer_stack) - 1:
                        break
                    else:
                        if len_dec_seq == 0:
                            dec_outputs.append(tmp)
                        else:
                            dec_outputs[i + 1] = torch.cat([dec_outputs[i + 1], tmp], 1)

                logits = self.tgt_word_prj(tmp.squeeze(1))

                chosen = torch.argmax(logits, -1)
                trg_seq = torch.cat([trg_seq, chosen.unsqueeze(-1)], -1)

        result = []
        for _ in trg_seq:
            result.append("")
            for elem in _:
                if elem.item() == Constants.EOS:
                    break
                elif elem.item() == Constants.SOS:
                    continue
                else:
                    result[-1] += de_vocab[elem.item()]
        return result
    """
    def translate_batch(self, de_vocab, src_seq, n_bm, max_token_seq_len=30):
        ''' Translation work in one batch '''
        device = src_seq.device
        def collate_active_info(src_seq, inst_idx_to_position_map, active_inst_idx_list):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(device)

            active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)

            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            return active_src_seq, active_inst_idx_to_position_map

        def beam_decode_step(inst_dec_beams, len_dec_seq, active_inst_idx_list, src_seq, \
                             inst_idx_to_position_map, n_bm):
            ''' Decode and update beam status, and then return active beam idx '''
            n_active_inst = len(inst_idx_to_position_map)

            dec_partial_seq = [inst_dec_beams[idx].get_current_state()
                               for idx in active_inst_idx_list if not inst_dec_beams[idx].done]
            dec_partial_seq = torch.stack(dec_partial_seq).to(device)
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)

            logits = self.forward(dec_partial_seq, src_seq)[:, -1, :] / Constants.T

            word_prob = F.log_softmax(logits, dim=1)
            word_prob = word_prob.view(n_active_inst, n_bm, -1)

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = []
            for inst_idx, inst_position in inst_idx_to_position_map.items():
                is_inst_complete = inst_dec_beams[inst_idx].advance(word_prob[inst_position])
                if not is_inst_complete:
                    active_inst_idx_list += [inst_idx]

            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for beam in inst_dec_beams:
                scores = beam.scores
                hyps = [beam.get_hypothesis(i, tensor=False) for i in range(beam.size)]
                normed_scores = [scores[i].item()/len(h) for i, h in enumerate(hyps)]

                idxs = np.argsort(normed_scores)[::-1]
                all_hyp.append([hyps[idx] for idx in idxs])
                all_scores.append([normed_scores[idx] for idx in idxs])

            return all_hyp, all_scores

        with torch.no_grad():
            #-- Repeat data for beam search
            n_inst, len_s = src_seq.size()
            src_seq = src_seq.repeat(1, n_bm).view(n_inst * n_bm, len_s)

            #-- Prepare beams
            inst_dec_beams = [Beam(n_bm, de_vocab, device=device) for _ in range(n_inst)]

            #-- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            #-- Decode
            for len_dec_seq in range(1, max_token_seq_len + 1):
                active_inst_idx_list = beam_decode_step(inst_dec_beams, len_dec_seq, active_inst_idx_list,
                                                        src_seq, inst_idx_to_position_map, n_bm)
                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                src_seq, inst_idx_to_position_map = collate_active_info(
                    src_seq, inst_idx_to_position_map, active_inst_idx_list)

            batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, n_bm)

            result = []
            for _ in batch_hyp:
                finished = False
                for r in _:
                    if len(r) >= 8:
                        result.append(r)
                        finished = True
                        break
                if not finished:
                    result.append(_[0])
            return result
    """


def get_inst_idx_to_tensor_position_map(inst_idx_list):
    ''' Indicate the position of an instance in a tensor. '''
    return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}


def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
    ''' Collect tensor parts associated to active instances. '''
    d_hs = beamed_tensor.size()[1:]
    new_shape = tuple([len(curr_active_inst_idx) * n_bm] + list(d_hs))
    beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
    beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
    beamed_tensor = beamed_tensor.view(*new_shape)

    return beamed_tensor
