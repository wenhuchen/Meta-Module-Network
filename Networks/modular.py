from .net_utils import FC, MLP, LayerNorm
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import Constants
import numpy as np
# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------


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


class MHAtt(nn.Module):
	def __init__(self, head_num, hidden_size, dropout, hidden_size_head):
		super(MHAtt, self).__init__()
		self.head_num = head_num
		self.hidden_size = hidden_size
		self.hidden_size_head = hidden_size_head
		self.linear_v = nn.Linear(hidden_size, hidden_size)
		self.linear_k = nn.Linear(hidden_size, hidden_size)
		self.linear_q = nn.Linear(hidden_size, hidden_size)
		self.linear_merge = nn.Linear(hidden_size, hidden_size)

		self.dropout = nn.Dropout(dropout, inplace=False)

	def forward(self, v, k, q, mask):
		n_batches = q.size(0)

		v = self.linear_v(v).view(
			n_batches,
			-1,
			self.head_num,
			self.hidden_size_head
		).transpose(1, 2)

		k = self.linear_k(k).view(
			n_batches,
			-1,
			self.head_num,
			self.hidden_size_head
		).transpose(1, 2)

		q = self.linear_q(q).view(
			n_batches,
			-1,
			self.head_num,
			self.hidden_size_head
		).transpose(1, 2)

		atted = self.att(v, k, q, mask)
		atted = atted.transpose(1, 2).contiguous().view(
			n_batches,
			-1,
			self.hidden_size
		)

		atted = self.linear_merge(atted)

		return atted

	def att(self, value, key, query, mask):
		d_k = query.size(-1)
		scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

		if mask is not None:
			scores = scores.masked_fill(mask, -1e9)

		att_map = F.softmax(scores, dim=-1)
		att_map = self.dropout(att_map)

		return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
	def __init__(self, hidden_size, ff_size, dropout):
		super(FFN, self).__init__()

		self.mlp = MLP(
			in_size=hidden_size,
			mid_size=ff_size,
			out_size=hidden_size,
			dropout_r=dropout,
			use_relu=True
		)

	def forward(self, x):
		return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
	def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head):
		super(SA, self).__init__()

		self.mhatt = MHAtt(head_num, hidden_size, dropout, hidden_size_head)
		self.ffn = FFN(hidden_size, ff_size, dropout)

		self.dropout1 = nn.Dropout(dropout, inplace=False)
		self.norm1 = LayerNorm(hidden_size)

		self.dropout2 = nn.Dropout(dropout, inplace=False)
		self.norm2 = LayerNorm(hidden_size)

	def forward(self, x, x_mask):
		output = self.mhatt(x, x, x, x_mask)
		dropout_output = self.dropout1(output)
		x = self.norm1(x + dropout_output)

		x = self.norm2(x + self.dropout2(
			self.ffn(x)
		))

		return x

# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------


class SGA(nn.Module):
	def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head):
		super(SGA, self).__init__()

		self.mhatt1 = MHAtt(head_num, hidden_size, dropout, hidden_size_head)
		self.mhatt2 = MHAtt(head_num, hidden_size, dropout, hidden_size_head)
		self.ffn = FFN(hidden_size, ff_size, dropout)

		self.dropout1 = nn.Dropout(dropout, inplace=False)
		self.norm1 = LayerNorm(hidden_size)

		self.dropout2 = nn.Dropout(dropout, inplace=False)
		self.norm2 = LayerNorm(hidden_size)

		self.dropout3 = nn.Dropout(dropout, inplace=False)
		self.norm3 = LayerNorm(hidden_size)

	def forward(self, x, y, x_mask, y_mask):
		x = self.norm1(x + self.dropout1(
			self.mhatt1(x, x, x, x_mask)
		))

		x = self.norm2(x + self.dropout2(
			self.mhatt2(y, y, x, y_mask)
		))

		x = self.norm3(x + self.dropout3(
			self.ffn(x)
		))

		return x


class GA(nn.Module):
	def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head):
		super(GA, self).__init__()

		self.mhatt = MHAtt(head_num, hidden_size, dropout, hidden_size_head)
		self.ffn = FFN(hidden_size, ff_size, dropout)

		self.dropout1 = nn.Dropout(dropout, inplace=False)
		self.norm1 = LayerNorm(hidden_size)

		self.dropout2 = nn.Dropout(dropout, inplace=False)
		self.norm2 = LayerNorm(hidden_size)

	def forward(self, x, y, y_mask, x_mask=None):
		if x_mask is None:
			intermediate = self.dropout1(self.mhatt(y, y, x, y_mask))
		else:
			intermediate = self.dropout1(self.mhatt(y, y, x, y_mask)) * x_mask.unsqueeze(-1)

		x = self.norm1(x + intermediate)
		x = self.norm2(x + self.dropout2(
			self.ffn(x)
		))

		return x

class Module(nn.Module):
	def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head):
		super(Module, self).__init__()
		self.self_attention = SA(hidden_size, head_num, ff_size, dropout, hidden_size_head)
		self.cross_attention = GA(hidden_size, head_num, ff_size, dropout, hidden_size_head)

	def forward(self, inputs, mask, vis_feat, vis_mask_tmp, program_masks, alpha):
		alpha = alpha.unsqueeze(-1)
		trans_mask = (1 - mask).unsqueeze(1).to(torch.bool)
		enc_output = self.self_attention(inputs, trans_mask)
		enc_output = self.cross_attention(enc_output, vis_feat, vis_mask_tmp)
		enc_output = enc_output * program_masks.unsqueeze(-1)
		return alpha * enc_output + (1 - alpha) * inputs


class ShallowModule(nn.Module):
	def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head, preprocessing=True):
		super(ShallowModule, self).__init__()
		#self.ffn = FFN(hidden_size, ff_size, dropout)
		#self.dropout = nn.Dropout(dropout, inplace=False)
		#self.norm = LayerNorm(hidden_size)
		self.cross_attention = GA(hidden_size, head_num, ff_size, dropout, hidden_size_head)

	def forward(self, inputs, vis_feat, vis_mask_tmp, program_masks):
		#inputs = self.norm(inputs + self.dropout(self.ffn(inputs)))
		enc_output = self.cross_attention(inputs, vis_feat, vis_mask_tmp)
		enc_output = enc_output * program_masks.unsqueeze(-1)
		return enc_output


class DeepModule(nn.Module):
	def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head, stacking):
		super(DeepModule, self).__init__()
		self.self_attention = SA(hidden_size, head_num, ff_size, dropout, hidden_size_head)
		self.cross_attention = nn.ModuleList([GA(hidden_size, head_num, ff_size, dropout, hidden_size_head)
											  for _ in range(stacking)])

	def forward(self, inputs, mask, vis_feat, vis_mask_tmp, program_masks, alpha):
		alpha = alpha.unsqueeze(-1)
		trans_mask = (1 - mask).unsqueeze(1).to(torch.bool)
		enc_output = self.self_attention(inputs, trans_mask)
		for cross_attention in self.cross_attention:
			enc_output = cross_attention(enc_output, vis_feat, vis_mask_tmp)
			enc_output = enc_output * program_masks.unsqueeze(-1)

		return alpha * enc_output + (1 - alpha) * inputs


class MCA_ED(nn.Module):
	def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head, layers):
		super(MCA_ED, self).__init__()

		self.enc_list = nn.ModuleList(
			[SA(hidden_size, head_num, ff_size, dropout, hidden_size_head) for _ in range(layers)])
		self.dec_list = nn.ModuleList(
			[SGA(hidden_size, head_num, ff_size, dropout, hidden_size_head) for _ in range(layers)])

	def forward(self, x, y, x_mask, y_mask):
		# Get hidden vector
		for enc in self.enc_list:
			x = enc(x, x_mask)

		for dec in self.dec_list:
			y = dec(y, x, y_mask, x_mask)

		return x, y


class AttFlat(nn.Module):
	def __init__(self, hidden_size, flat_mlp_size, flat_glimpse, flat_out_size, dropout):
		super(AttFlat, self).__init__()

		self.flat_glimpse = flat_glimpse
		self.mlp = MLP(
			in_size=hidden_size,
			mid_size=flat_mlp_size,
			out_size=flat_glimpse,
			dropout_r=dropout,
			use_relu=True
		)

		self.linear_merge = nn.Linear(hidden_size * flat_glimpse, flat_out_size)

	def forward(self, x, x_mask):
		att = self.mlp(x)
		att = att.masked_fill(
			x_mask.squeeze(1).squeeze(1).unsqueeze(2),
			-1e9
		)
		att = F.softmax(att, dim=1)

		att_list = []
		for i in range(self.flat_glimpse):
			att_list.append(
				torch.sum(att[:, :, i: i + 1] * x, dim=1)
			)

		x_atted = torch.cat(att_list, dim=1)
		x_atted = self.linear_merge(x_atted)

		return x_atted

	def forward_step(self, x, x_mask):
		att = self.mlp(x)
		att = att.masked_fill(x_mask.unsqueeze(-1), -1e9)
		att = F.softmax(att, dim=1)

		att_list = []
		for i in range(self.flat_glimpse):
			att_list.append(
				torch.sum(att[:, :, i: i + 1] * x, dim=1)
			)

		x_atted = torch.cat(att_list, dim=1)
		x_atted = self.linear_merge(x_atted)

		return x_atted


class JointEmbedding(nn.Module):
	def __init__(self, vocab_size, hidden_dim):
		super(JointEmbedding, self).__init__()
		self.embedding = nn.Embedding(vocab_size, 300, padding_idx=Constants.PAD)
		self.ques_proj = nn.Linear(300, hidden_dim)
		self.prog_proj = nn.Linear(300, hidden_dim // 8)

	def load_embedding(self, pre_trained):
		self.embedding.weight.data.copy_(torch.from_numpy(np.load(pre_trained)))

	def encode_ques(self, ques):
		return self.ques_proj(self.embedding(ques))

	def encode_prog(self, prog):
		batch_size = prog.size(0)
		length = prog.size(1)
		return self.prog_proj(self.embedding(prog)).view(batch_size, length, -1)


class VisualEncoder(nn.Module):
	def __init__(self, hidden_dim, n_head, pre_layers, dropout):
		super(VisualEncoder, self).__init__()
		self.ques_encoder = nn.ModuleList(
			[SA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
			 for _ in range(pre_layers)])
		self.vis_encoder = nn.ModuleList(
			[SGA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
			 for _ in range(pre_layers)])

	def forward(self, ques_input, vis_feat, ques_mask_tmp, vis_mask_tmp, ques_masks, vis_mask):
		for enc in self.ques_encoder:
			ques_input = enc(ques_input, ques_mask_tmp)
			ques_input *= ques_masks.unsqueeze(-1)
		# Encoding the visual feature
		for enc in self.vis_encoder:
			vis_feat = enc(vis_feat, ques_input, vis_mask_tmp, ques_mask_tmp)
			vis_feat *= vis_mask.unsqueeze(-1)
		return vis_feat


class MCAN(nn.Module):
	def __init__(self, vocab_size, answer_size, visual_dim, coordinate_dim, hidden_dim, n_head, n_layers, stacking, dropout):
		super(MCAN, self).__init__()

		self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=300)
		self.img_feat_linear = nn.Linear(visual_dim, hidden_dim)
		self.coordinate_proj = nn.Linear(coordinate_dim, hidden_dim)

		self.lstm = nn.LSTM(
			input_size=300,
			hidden_size=hidden_dim,
			num_layers=1,
			batch_first=True
		)
		self.backbone = MCA_ED(hidden_size=hidden_dim, head_num=n_head, ff_size=4 * hidden_dim,
							   dropout=dropout, hidden_size_head=hidden_dim // n_head, layers=stacking)

		self.attflat_img = AttFlat(hidden_size=hidden_dim, flat_mlp_size=hidden_dim, flat_glimpse=1,
								   flat_out_size=2 * hidden_dim, dropout=dropout)
		self.attflat_lang = AttFlat(hidden_size=hidden_dim, flat_mlp_size=hidden_dim, flat_glimpse=1,
									flat_out_size=2 * hidden_dim, dropout=dropout)

		self.proj_norm = LayerNorm(2 * hidden_dim)
		self.proj = nn.Linear(2 * hidden_dim, answer_size)

	def forward(self, ques, ques_masks, program, program_masks, transition_masks, activate_masks, vis_feat, box_feat, vis_mask, index, depth):

		# Pre-process Language Feature
		lang_feat = self.embedding(ques)
		self.lstm.flatten_parameters()
		lang_feat, _ = self.lstm(lang_feat)

		# Pre-process Image Feature
		img_feat = self.img_feat_linear(vis_feat) + self.coordinate_proj(box_feat)

		# Generate masking
		ques_masks = self.make_mask(ques.unsqueeze(2))
		vis_mask = self.make_mask(vis_feat)

		# Backbone Framework
		lang_feat, img_feat = self.backbone(lang_feat, img_feat, ques_masks, vis_mask)

		lang_feat = self.attflat_lang(lang_feat, ques_masks)

		img_feat = self.attflat_img(img_feat, vis_mask)

		proj_feat = lang_feat + img_feat
		proj_feat = self.proj_norm(proj_feat)
		logits = self.proj(proj_feat)

		return logits

	# Masking
	def make_mask(self, feature):
		return (torch.sum(torch.abs(feature), dim=-1) == 0).unsqueeze(1).unsqueeze(2)


class TreeTransformer(nn.Module):
	def __init__(self, vocab_size, answer_size, visual_dim, hidden_dim, coordinate_dim,
				 n_head, n_layers, stacking, dropout, intermediate_dim, pre_layers, intermediate_layer):
		super(TreeTransformer, self).__init__()
		# The question encoder
		# self.embedding = JointEmbedding(vocab_size, hidden_dim)
		self.embedding = nn.Embedding(vocab_size, 300, padding_idx=Constants.PAD)
		self.ques_proj = nn.Linear(300, hidden_dim)
		self.prog_proj = nn.Linear(300, hidden_dim // 8)

		self.ques_pos_emb = PositionalEmbedding(hidden_dim)
		self.intermediate_layer = intermediate_layer
		# The visual encoder
		self.vis_proj = nn.Linear(visual_dim, hidden_dim)
		self.coordinate_proj = nn.Linear(coordinate_dim, hidden_dim)
		# self.vis_encoder = VisualEncoder(hidden_dim, n_head, pre_layers, dropout)
		self.ques_encoder = nn.ModuleList(
			[SA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
			 for _ in range(pre_layers)])
		self.vis_encoder = nn.ModuleList(
			[SGA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
			 for _ in range(pre_layers)])

		# The program decoder
		self.num_regions = intermediate_dim
		self.pos_emb = nn.Embedding(self.num_regions, hidden_dim)
		self.idx_predictor = nn.Linear(hidden_dim, self.num_regions)

		# The self attention module and cross attention module
		self.prog_encoder = nn.ModuleList(
			[SA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head) for _ in range(n_layers)])
		self.cross_decoder = nn.ModuleList(
			[GA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head) for _ in range(n_layers)])

		# Projection layer to retrieve final answer
		self.proj = nn.Linear(hidden_dim, answer_size)

	def forward(self, ques, ques_masks, program, program_masks, transition_masks, activate_masks, vis_feat, box_feat, vis_mask, index, depth):
		batch_size = ques.size(0)
		idx = torch.arange(vis_feat.size(1)).unsqueeze(0).repeat(batch_size, 1).to(ques.device)
		vis_feat = self.vis_proj(vis_feat) + self.coordinate_proj(box_feat) + self.pos_emb(idx)

		vis_mask_tmp = (1 - vis_mask).unsqueeze(1).unsqueeze(2).to(torch.bool)
		program_mask_tmp = (1 - program_masks).unsqueeze(1).unsqueeze(2).to(torch.bool)
		ques_mask_tmp = (1 - ques_masks).unsqueeze(1).unsqueeze(2).to(torch.bool)

		# Encoding the question with self-attention
		ques_input = self.ques_proj(self.embedding(ques)) + self.ques_pos_emb(ques)
		for enc in self.ques_encoder:
			ques_input = enc(ques_input, ques_mask_tmp)
			ques_input *= ques_masks.unsqueeze(-1)
		# Encoding the visual feature
		for enc in self.vis_encoder:
			vis_feat = enc(vis_feat, ques_input, vis_mask_tmp, ques_mask_tmp)
			vis_feat *= vis_mask.unsqueeze(-1)

		# Decoding the programs
		buffers = [self.prog_proj(self.embedding(program)).view(batch_size, program.size(1), -1)]
		transition_masks = transition_masks.transpose(0, 1)
		for prog, cross, mask in zip(self.prog_encoder, self.cross_decoder, transition_masks):
			enc_output = prog(buffers[-1], (1 - mask).unsqueeze(1).to(torch.bool))
			enc_output = cross(enc_output, vis_feat, vis_mask_tmp)
			buffers.append(enc_output * program_masks.unsqueeze(-1))
		buffers.pop(0)

		# Concatenating and collecting all the outputs
		if self.intermediate_layer:
			enc_outputs = torch.cat([_.unsqueeze(2) for _ in buffers], 2)
			enc_output = torch.gather(enc_outputs, 2, depth.unsqueeze(-1).unsqueeze(-1).repeat(1,
																							   1, 1, enc_outputs.size(-1))).squeeze(2)
		else:
			enc_output = buffers[-1]
		# Predict the intermediate results
		pre_logits = self.idx_predictor(enc_output)

		lang_feat = torch.gather(enc_output, 1, index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, enc_output.size(-1)))
		logits = self.proj(lang_feat.view(batch_size, -1))
		return pre_logits, logits


class TreeTransformerSparse(nn.Module):
	def __init__(self, vocab_size, answer_size, visual_dim, hidden_dim, coordinate_dim,
				 n_head, n_layers, stacking, dropout, intermediate_dim, pre_layers, intermediate_layer):
		super(TreeTransformerSparse, self).__init__()
		# The question encoder
		self.embedding = nn.Embedding(vocab_size, 300, padding_idx=Constants.PAD)
		self.ques_proj = nn.Linear(300, hidden_dim)
		self.prog_proj = nn.Linear(300, hidden_dim // 8)

		self.ques_pos_emb = PositionalEmbedding(hidden_dim)
		self.intermediate_layer = intermediate_layer
		# The visual encoder
		self.vis_proj = nn.Linear(visual_dim, hidden_dim)
		self.coordinate_proj = nn.Linear(coordinate_dim, hidden_dim)
		# self.vis_encoder = VisualEncoder(hidden_dim, n_head, pre_layers, dropout)
		self.ques_encoder = nn.ModuleList(
			[SA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
			 for _ in range(pre_layers)])
		self.vis_encoder = nn.ModuleList(
			[SGA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
			 for _ in range(pre_layers)])

		# The program decoder
		self.num_regions = intermediate_dim
		self.pos_emb = nn.Embedding(self.num_regions, hidden_dim)

		# The self attention module and cross attention module
		self.layers = Module(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)

		# Projection layer to retrieve final answer
		self.idx_predictor = nn.Linear(hidden_dim, self.num_regions)
		self.proj = nn.Linear(hidden_dim, answer_size)

	def forward(self, ques, ques_masks, program, program_masks, transition_masks, activate_masks, vis_feat, box_feat, vis_mask, index, depth):
		batch_size = ques.size(0)
		length = program.size(1)
		idx = torch.arange(vis_feat.size(1)).unsqueeze(0).repeat(batch_size, 1).to(ques.device)
		vis_feat = self.vis_proj(vis_feat) + self.coordinate_proj(box_feat) + self.pos_emb(idx)

		vis_mask_tmp = (1 - vis_mask).unsqueeze(1).unsqueeze(2).to(torch.bool)
		program_mask_tmp = (1 - program_masks).unsqueeze(1).unsqueeze(2).to(torch.bool)
		ques_mask_tmp = (1 - ques_masks).unsqueeze(1).unsqueeze(2).to(torch.bool)

		# Encoding the question with self-attention
		ques_input = self.ques_proj(self.embedding(ques)) + self.ques_pos_emb(ques)
		for enc in self.ques_encoder:
			ques_input = enc(ques_input, ques_mask_tmp)
			ques_input *= ques_masks.unsqueeze(-1)
		# Encoding the visual feature
		for enc in self.vis_encoder:
			vis_feat = enc(vis_feat, ques_input, vis_mask_tmp, ques_mask_tmp)
			vis_feat *= vis_mask.unsqueeze(-1)

		enc_output = self.prog_proj(self.embedding(program)).view(
			batch_size, program.size(1), -1) * program_masks.unsqueeze(-1)

		transition_masks = transition_masks.transpose(0, 1)
		activate_masks = activate_masks.transpose(0, 1)

		for trans_mask, active_mask in zip(transition_masks, activate_masks):
			enc_output = self.layers(enc_output, trans_mask, vis_feat, vis_mask_tmp, program_masks, active_mask)

		# Predict the intermediate results
		pre_logits = self.idx_predictor(enc_output)

		lang_feat = torch.gather(enc_output, 1, index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, enc_output.size(-1)))
		logits = self.proj(lang_feat.view(batch_size, -1))

		return pre_logits, logits


class TreeTransformerDeepSparse(nn.Module):
	def __init__(self, vocab_size, answer_size, visual_dim, hidden_dim, coordinate_dim,
				 n_head, n_layers, stacking, dropout, intermediate_dim, pre_layers, intermediate_layer):
		super(TreeTransformerDeepSparse, self).__init__()
		# The question encoder
		self.embedding = nn.Embedding(vocab_size, 300, padding_idx=Constants.PAD)
		self.ques_proj = nn.Linear(300, hidden_dim)
		self.prog_proj = nn.Linear(300, hidden_dim // 8)

		self.ques_pos_emb = PositionalEmbedding(hidden_dim)
		self.intermediate_layer = intermediate_layer
		# The visual encoder
		self.vis_proj = nn.Linear(visual_dim, hidden_dim)
		self.coordinate_proj = nn.Linear(coordinate_dim, hidden_dim)
		# self.vis_encoder = VisualEncoder(hidden_dim, n_head, pre_layers, dropout)
		self.ques_encoder = nn.ModuleList(
			[SA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
			 for _ in range(pre_layers)])
		self.vis_encoder = nn.ModuleList(
			[SGA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
			 for _ in range(pre_layers)])

		# The program decoder
		self.num_regions = intermediate_dim
		self.pos_emb = nn.Embedding(self.num_regions, hidden_dim)

		# The self attention module and cross attention module
		self.layers = DeepModule(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head, stacking)

		# Projection layer to retrieve final answer
		self.idx_predictor = nn.Linear(hidden_dim, self.num_regions)
		self.proj = nn.Linear(hidden_dim, answer_size)

	def forward(self, ques, ques_masks, program, program_masks, transition_masks, activate_masks, vis_feat, box_feat, vis_mask, index, depth):
		batch_size = ques.size(0)
		length = program.size(1)
		idx = torch.arange(vis_feat.size(1)).unsqueeze(0).repeat(batch_size, 1).to(ques.device)
		vis_feat = self.vis_proj(vis_feat) + self.coordinate_proj(box_feat) + self.pos_emb(idx)

		vis_mask_tmp = (1 - vis_mask).unsqueeze(1).unsqueeze(2).to(torch.bool)
		program_mask_tmp = (1 - program_masks).unsqueeze(1).unsqueeze(2).to(torch.bool)
		ques_mask_tmp = (1 - ques_masks).unsqueeze(1).unsqueeze(2).to(torch.bool)

		# Encoding the question with self-attention
		ques_input = self.ques_proj(self.embedding(ques)) + self.ques_pos_emb(ques)
		for enc in self.ques_encoder:
			ques_input = enc(ques_input, ques_mask_tmp)
			ques_input *= ques_masks.unsqueeze(-1)
		# Encoding the visual feature
		for enc in self.vis_encoder:
			vis_feat = enc(vis_feat, ques_input, vis_mask_tmp, ques_mask_tmp)
			vis_feat *= vis_mask.unsqueeze(-1)

		enc_output = self.prog_proj(self.embedding(program)).view(
			batch_size, program.size(1), -1) * program_masks.unsqueeze(-1)

		transition_masks = transition_masks.transpose(0, 1)
		activate_masks = activate_masks.transpose(0, 1)

		for trans_mask, active_mask in zip(transition_masks, activate_masks):
			enc_output = self.layers(enc_output, trans_mask, vis_feat, vis_mask_tmp, program_masks, active_mask)

		# Predict the intermediate results
		pre_logits = self.idx_predictor(enc_output)

		lang_feat = torch.gather(enc_output, 1, index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, enc_output.size(-1)))
		logits = self.proj(lang_feat.view(batch_size, -1))

		return pre_logits, logits


class TreeTransformerSparsePost(nn.Module):
	def __init__(self, vocab_size, answer_size, visual_dim, hidden_dim, coordinate_dim,
				 n_head, n_layers, stacking, dropout, intermediate_dim, pre_layers, intermediate_layer):
		super(TreeTransformerSparsePost, self).__init__()
		# The question encoder
		self.embedding = nn.Embedding(vocab_size, 300, padding_idx=Constants.PAD)
		self.ques_proj = nn.Linear(300, hidden_dim)
		self.prog_proj = nn.Linear(300, hidden_dim // 8)

		self.ques_pos_emb = PositionalEmbedding(hidden_dim)
		self.intermediate_layer = intermediate_layer
		# The visual encoder
		self.vis_proj = nn.Linear(visual_dim, hidden_dim)
		self.coordinate_proj = nn.Linear(coordinate_dim, hidden_dim)
		# self.vis_encoder = VisualEncoder(hidden_dim, n_head, pre_layers, dropout)
		self.ques_encoder = nn.ModuleList(
			[SA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
			 for _ in range(pre_layers)])
		self.vis_encoder = nn.ModuleList(
			[SGA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
			 for _ in range(pre_layers)])

		# The program decoder
		self.num_regions = intermediate_dim
		self.pos_emb = nn.Embedding(self.num_regions, hidden_dim)
		self.idx_predictor = nn.Linear(hidden_dim, self.num_regions)

		# The self attention module beforehand
		self.post = nn.ModuleList([Module(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
									for _ in range(stacking)])

		# The self attention module and cross attention module
		self.module = Module(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)

		# Projection layer to retrieve final answer
		self.proj = nn.Linear(hidden_dim, answer_size)

	def forward(self, ques, ques_masks, program, program_masks, transition_masks, activate_masks, vis_feat, box_feat, vis_mask, index, depth):
		batch_size = ques.size(0)
		length = program.size(1)
		idx = torch.arange(vis_feat.size(1)).unsqueeze(0).repeat(batch_size, 1).to(ques.device)
		vis_feat = self.vis_proj(vis_feat) + self.coordinate_proj(box_feat) + self.pos_emb(idx)

		vis_mask_tmp = (1 - vis_mask).unsqueeze(1).unsqueeze(2).to(torch.bool)
		program_mask_tmp = (1 - program_masks).unsqueeze(1).unsqueeze(2).to(torch.bool)
		ques_mask_tmp = (1 - ques_masks).unsqueeze(1).unsqueeze(2).to(torch.bool)

		# Encoding the question with self-attention
		ques_input = self.ques_proj(self.embedding(ques)) + self.ques_pos_emb(ques)
		for enc in self.ques_encoder:
			ques_input = enc(ques_input, ques_mask_tmp)
			ques_input *= ques_masks.unsqueeze(-1)
		# Encoding the visual feature
		for enc in self.vis_encoder:
			vis_feat = enc(vis_feat, ques_input, vis_mask_tmp, ques_mask_tmp)
			vis_feat *= vis_mask.unsqueeze(-1)

		enc_output = self.prog_proj(self.embedding(program)).view(batch_size, program.size(1), -1)
		transition_masks = transition_masks.transpose(0, 1)
		activate_masks = activate_masks.transpose(0, 1)

		# Build the structure into the transformer
		for trans_mask, active_mask in zip(transition_masks, activate_masks):
			enc_output = self.module(enc_output, trans_mask, vis_feat, vis_mask_tmp, program_masks, active_mask)

		# Pre-process the encoder input
		trans_mask = torch.eye(length).unsqueeze(0).repeat(batch_size, 1, 1).to(ques.device)
		active_mask = torch.FloatTensor(batch_size, length).fill_(1).to(ques.device)

		for layer in self.post:
			enc_output = layer(enc_output, trans_mask, vis_feat, vis_mask_tmp, program_masks, active_mask)

		# Predict the intermediate results
		pre_logits = self.idx_predictor(enc_output)

		lang_feat = torch.gather(enc_output, 1, index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, enc_output.size(-1)))
		logits = self.proj(lang_feat.view(batch_size, -1))

		return pre_logits, logits


class TreeTransformerSparsePostv2(nn.Module):
	def __init__(self, vocab_size, answer_size, visual_dim, hidden_dim, coordinate_dim,
				 n_head, n_layers, stacking, dropout, intermediate_dim, pre_layers, intermediate_layer):
		super(TreeTransformerSparsePostv2, self).__init__()
		# The question encoder
		self.embedding = nn.Embedding(vocab_size, 300, padding_idx=Constants.PAD)
		self.ques_proj = nn.Linear(300, hidden_dim)
		self.prog_proj = nn.Linear(300, hidden_dim // 8)

		self.ques_pos_emb = PositionalEmbedding(hidden_dim)
		self.intermediate_layer = intermediate_layer
		# The visual encoder
		self.vis_proj = nn.Linear(visual_dim, hidden_dim)
		self.coordinate_proj = nn.Linear(coordinate_dim, hidden_dim)
		# self.vis_encoder = VisualEncoder(hidden_dim, n_head, pre_layers, dropout)
		self.ques_encoder = nn.ModuleList(
			[SA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
			 for _ in range(pre_layers)])
		self.vis_encoder = nn.ModuleList(
			[SGA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
			 for _ in range(pre_layers)])

		# The program decoder
		self.num_regions = intermediate_dim
		self.pos_emb = nn.Embedding(self.num_regions, hidden_dim)
		self.idx_predictor = nn.Linear(hidden_dim, self.num_regions)

		# The self attention module beforehand
		self.post = nn.ModuleList([ShallowModule(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
									for _ in range(stacking)])

		# The self attention module and cross attention module
		self.module = Module(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)

		# Projection layer to retrieve final answer
		self.proj = nn.Linear(hidden_dim, answer_size)

	def forward(self, ques, ques_masks, program, program_masks, transition_masks, activate_masks, vis_feat, box_feat, vis_mask, index, depth):
		batch_size = ques.size(0)
		length = program.size(1)
		idx = torch.arange(vis_feat.size(1)).unsqueeze(0).repeat(batch_size, 1).to(ques.device)
		vis_feat = self.vis_proj(vis_feat) + self.coordinate_proj(box_feat) + self.pos_emb(idx)

		vis_mask_tmp = (1 - vis_mask).unsqueeze(1).unsqueeze(2).to(torch.bool)
		program_mask_tmp = (1 - program_masks).unsqueeze(1).unsqueeze(2).to(torch.bool)
		ques_mask_tmp = (1 - ques_masks).unsqueeze(1).unsqueeze(2).to(torch.bool)

		# Encoding the question with self-attention
		ques_input = self.ques_proj(self.embedding(ques)) + self.ques_pos_emb(ques)
		for enc in self.ques_encoder:
			ques_input = enc(ques_input, ques_mask_tmp)
			ques_input *= ques_masks.unsqueeze(-1)
		# Encoding the visual feature
		for enc in self.vis_encoder:
			vis_feat = enc(vis_feat, ques_input, vis_mask_tmp, ques_mask_tmp)
			vis_feat *= vis_mask.unsqueeze(-1)

		enc_output = self.prog_proj(self.embedding(program)).view(batch_size, program.size(1), -1)
		transition_masks = transition_masks.transpose(0, 1)
		activate_masks = activate_masks.transpose(0, 1)

		# Build the structure into the transformer
		for trans_mask, active_mask in zip(transition_masks, activate_masks):
			enc_output = self.module(enc_output, trans_mask, vis_feat, vis_mask_tmp, program_masks, active_mask)

		# Post-Processing the encoder output
		for layer in self.post:
			enc_output = layer(enc_output, vis_feat, vis_mask_tmp, program_masks)

		# Predict the intermediate results
		pre_logits = self.idx_predictor(enc_output)

		lang_feat = torch.gather(enc_output, 1, index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, enc_output.size(-1)))
		logits = self.proj(lang_feat.view(batch_size, -1))

		return pre_logits, logits


class OnlyVision(nn.Module):
	def __init__(self, vocab_size, answer_size, visual_dim, hidden_dim, coordinate_dim,
				 n_head, n_layers, stacking, dropout, intermediate_dim, pre_layers, intermediate_layer):
		super(OnlyVision, self).__init__()
		# The question encoder
		self.embedding = nn.Embedding(vocab_size, 300, padding_idx=Constants.PAD)
		self.prog_embedding = nn.Embedding(vocab_size, 300, padding_idx=Constants.PAD)

		self.ques_encoder = nn.ModuleList(
			[SA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head) for _ in range(pre_layers)])
		self.ques_proj = nn.Linear(300, hidden_dim)
		self.ques_pos_emb = PositionalEmbedding(hidden_dim)
		self.intermediate_layer = intermediate_layer
		# The visual encoder
		self.vis_proj = nn.Linear(visual_dim, hidden_dim)
		self.coordinate_proj = nn.Linear(coordinate_dim, hidden_dim)
		self.vis_encoder = nn.ModuleList(
			[SGA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head) for _ in range(pre_layers)])

		# The program decoder
		self.num_regions = intermediate_dim
		self.pos_emb = nn.Embedding(self.num_regions, hidden_dim)
		self.idx_predictor = nn.Linear(hidden_dim, self.num_regions)

		self.prog_proj = nn.Linear(300, hidden_dim // 8)

		# The self attention module and cross attention module
		self.prog_encoder = nn.ModuleList(
			[SA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head) for _ in range(n_layers)])
		self.cross_decoder = nn.ModuleList(
			[GA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head) for _ in range(n_layers)])

		self.attflat_img = AttFlat(hidden_size=hidden_dim, flat_mlp_size=hidden_dim, flat_glimpse=1,
								   flat_out_size=2 * hidden_dim, dropout=dropout)

		self.proj_norm = LayerNorm(2 * hidden_dim)
		self.proj = nn.Linear(2 * hidden_dim, answer_size)

	def forward(self, ques, ques_masks, program, program_masks, transition_masks, activate_masks, vis_feat, box_feat, vis_mask, index, depth):
		batch_size = ques.size(0)
		idx = torch.arange(vis_feat.size(1)).unsqueeze(0).repeat(batch_size, 1).to(ques.device)
		vis_feat = self.vis_proj(vis_feat) + self.coordinate_proj(box_feat) + self.pos_emb(idx)

		vis_mask_tmp = (1 - vis_mask).unsqueeze(1).unsqueeze(2).to(torch.bool)
		program_mask_tmp = (1 - program_masks).unsqueeze(1).unsqueeze(2).to(torch.bool)
		ques_mask_tmp = (1 - ques_masks).unsqueeze(1).unsqueeze(2).to(torch.bool)

		# Encoding the question with self-attention
		ques_input = self.ques_proj(self.embedding(ques)) + self.ques_pos_emb(ques)
		for enc in self.ques_encoder:
			ques_input = enc(ques_input, ques_mask_tmp)
			ques_input *= ques_masks.unsqueeze(-1)
		# Encoding the visual feature
		for enc in self.vis_encoder:
			vis_feat = enc(vis_feat, ques_input, vis_mask_tmp, ques_mask_tmp)
			vis_feat *= vis_mask.unsqueeze(-1)

		vis_feat = self.attflat_img(vis_feat, vis_mask_tmp)
		logits = self.proj(self.proj_norm(vis_feat))

		return logits


class TreeTransformerConcept(nn.Module):
	def __init__(self, q_vocab_size, p_vocab_size, c_vocab_size, answer_size, visual_dim, hidden_dim, coordinate_dim,
				 n_head, n_layers, stacking, dropout, intermediate_dim, pre_layers):
		super(TreeTransformerConcept, self).__init__()
		# The concept encoder
		self.concept_emb = nn.Embedding(c_vocab_size, 300, padding_idx=Constants.PAD)
		self.concept_proj = nn.Linear(300, hidden_dim)
		self.concept_encoder = SA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
		self.vis_concepter = GA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)

		# The question encoder
		self.ques_emb = nn.Embedding(q_vocab_size, 300, padding_idx=Constants.PAD)
		self.ques_proj = nn.Linear(300, hidden_dim)
		self.ques_encoder = nn.ModuleList(
			[SA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head) for _ in range(pre_layers)])
		self.ques_pos_emb = PositionalEmbedding(hidden_dim)
		# The visual encoder
		self.vis_proj = nn.Linear(visual_dim, hidden_dim)
		self.coordinate_proj = nn.Linear(coordinate_dim, hidden_dim)
		self.vis_encoder = nn.ModuleList(
			[SGA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head) for _ in range(pre_layers)])

		# The program decoder
		self.num_regions = intermediate_dim
		self.pos_emb = nn.Embedding(self.num_regions, hidden_dim)
		self.idx_predictor = nn.Linear(hidden_dim, self.num_regions)
		self.prog_emb = nn.Embedding(p_vocab_size, hidden_dim // 8, padding_idx=Constants.PAD)

		# The self attention module and cross attention module
		self.prog_encoder = nn.ModuleList(
			[SA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head) for _ in range(n_layers)])
		self.cross_decoder = nn.ModuleList(
			[GA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head) for _ in range(n_layers)])

		# Projection layer to retrieve final answer
		self.proj = nn.Linear(hidden_dim, answer_size)

	def forward(self, ques, ques_masks, program, program_masks, concepts, concept_mask, transition_masks, vis_feat, box_feat, vis_mask, index, depth):
		batch_size = ques.size(0)
		idx = torch.arange(vis_feat.size(1)).unsqueeze(0).repeat(batch_size, 1).to(ques.device)
		vis_feat = self.vis_proj(vis_feat) + self.coordinate_proj(box_feat) + self.pos_emb(idx)

		vis_mask_tmp = (1 - vis_mask).unsqueeze(1).unsqueeze(2).to(torch.bool)
		program_mask_tmp = (1 - program_masks).unsqueeze(1).unsqueeze(2).to(torch.bool)
		ques_mask_tmp = (1 - ques_masks).unsqueeze(1).unsqueeze(2).to(torch.bool)
		concept_mask_tmp = (1 - concept_mask).unsqueeze(1).unsqueeze(2).to(torch.bool)

		concepts = self.concept_proj(self.concept_emb(concepts))
		concepts = self.concept_encoder(concepts, concept_mask_tmp)
		vis_feat = self.vis_concepter(vis_feat, concepts, concept_mask_tmp)

		# Encoding the question with self-attention
		ques_input = self.ques_proj(self.ques_emb(ques)) + self.ques_pos_emb(ques)
		for enc in self.ques_encoder:
			ques_input = enc(ques_input, ques_mask_tmp)
			ques_input *= ques_masks.unsqueeze(-1)
		# Encoding the visual feature
		for enc in self.vis_encoder:
			vis_feat = enc(vis_feat, ques_input, vis_mask_tmp, ques_mask_tmp)
			vis_feat *= vis_mask.unsqueeze(-1)

		# Decoding the programs
		buffers = [self.prog_emb(program).view(batch_size, program.size(1), -1)]
		transition_masks = transition_masks.transpose(0, 1)
		for prog, cross, mask in zip(self.prog_encoder, self.cross_decoder, transition_masks):
			enc_output = prog(buffers[-1], (1 - mask).unsqueeze(1).to(torch.bool))
			enc_output = cross(enc_output, vis_feat, vis_mask_tmp)
			buffers.append(enc_output * program_masks.unsqueeze(-1))
		buffers.pop(0)

		# Concatenating and collecting all the outputs
		# enc_outputs = torch.cat([_.unsqueeze(2) for _ in buffers], 2)
		# enc_output = torch.gather(enc_outputs, 2, depth.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, enc_outputs.size(-1))).squeeze(2)
		enc_output = buffers[-1]
		# Predict the intermediate results
		pre_logits = self.idx_predictor(enc_output)

		lang_feat = torch.gather(enc_output, 1, index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, enc_output.size(-1)))
		logits = self.proj(lang_feat.view(batch_size, -1))

		return pre_logits, logits


class ConceptNet(nn.Module):
	def __init__(self, input_dim, vocab_size, dropout):
		super(ConceptNet, self).__init__()
		self.input_dim = input_dim
		self.n_head = 4
		self.attention = SA(input_dim, self.n_head, 4 * input_dim, dropout, input_dim // self.n_head)
		self.network = SimpleClassifier(input_dim, 1024, vocab_size, dropout)

	def forward(self, vis_feat, vis_mask):
		vis_mask_tmp = (1 - vis_mask).unsqueeze(1).unsqueeze(2).to(torch.bool)
		vis_feat = self.attention(vis_feat, vis_mask_tmp)
		prob = self.network(vis_feat)
		logits = torch.max(prob, 1)[0]
		return logits
