import json
import re
import argparse
import time
from collections import Counter
from multiprocessing import Pool
import multiprocessing
from nltk.tokenize import word_tokenize
import random
import operator
import torch.optim as optim
from torch import nn
import torch
from torch.autograd import Variable
from Networks.seq2seq import TransformerDecoder
import Constants
import numpy as np
from torch.utils.data import Dataset, DataLoader
import resource
import os
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (40000, rlimit[1]))
device = torch.device('cuda')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_preprocess', default=False, action="store_true",
                        help="whether to train or test the model")
    parser.add_argument('--do_train', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_trainval_unbiased', default=False, action="store_true",
                        help="whether to train or test the model")
    parser.add_argument('--do_val', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_test', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_testdev', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_submission', default=False, action="store_true",
                        help="whether to train or test the model")
    parser.add_argument('--batch_size', default=1024, type=int, help="The batch size during training")
    parser.add_argument('--hidden_dim', default=128, type=int, help="The hidden size of the state")
    parser.add_argument('--version', default='', type=str, help="The hidden size of the state")
    parser.add_argument('--beam_size', default=1, type=int, help="The hidden size of the state")
    parser.add_argument('--load_from', type=str, default="", help="whether to train or test the model")
    parser.add_argument('--max_len', default=100, type=int, help="The hidden size of the state")
    parser.add_argument('--debug', default=False, action="store_true", help="Whether to debug it")
    parser.add_argument('--max_epoch', default=10, type=int, help="The hidden size of the state")
    parser.add_argument('--output', default="nl2prog/", type=str, help="The hidden size of the state")
    parser.add_argument('--meta', default="meta_info/", type=str, help="The hidden size of the state")
    args = parser.parse_args()
    return args


args = parse_opt()


class GQA(Dataset):
    def __init__(self, vocab, split, folder='../gqa_bottom_up_features/'):
        self.vocab = vocab
        self.max_src = 30
        self.max_trg = 80
        self.split = split
        self.num_regions = 32
        self.folder = folder
        if split == 'train':
            with open('{}/train_pairs.json'.format(args.output)) as f:
                self.data = json.load(f)
        elif split == 'trainval_unbiased':
            with open('{}/train_balanced_pairs.json'.format(args.output)) as f:
                self.data = json.load(f)
        elif split == 'testdev':
            with open('{}/testdev_pairs.json'.format(args.output)) as f:
                self.data = json.load(f)
        elif split == 'submission':
            with open('{}/submission_pairs.json'.format(args.output)) as f:
                self.data = json.load(f)
        else:
            raise ValueError('unseen option')
        print("finished loading the data, totally {} instances".format(len(self.data)))

    def __getitem__(self, index):
        entry = self.data[index]

        question = entry[0]
        x = [self.vocab.get(_, Constants.UNK) for _ in question]

        output = entry[1]
        y = [self.vocab.get(_, Constants.UNK) for _ in output]

        if len(x) >= self.max_src:
            x = x[:self.max_src]

        src = np.array(x + [Constants.EOS] + [Constants.PAD] * (self.max_src - len(x)), 'int64')

        if len(y) >= self.max_trg:
            y = y[:self.max_trg]

        trg_inp = np.array([Constants.SOS] + y + [Constants.EOS] + [Constants.PAD] * (self.max_trg - len(y)), 'int64')
        trg_gt = np.array(y + [Constants.EOS] + [Constants.PAD] * (self.max_trg + 1 - len(y)), 'int64')

        image_id = entry[2]
        if not image_id.startswith('n'):
            if len(image_id) < 7:
                image_id = "0" * (7 - len(image_id)) + image_id

        bottom_up = np.load(os.path.join(self.folder, 'gqa_{}.npz'.format(image_id)))
        #object_feat = bottom_up['features'][:self.num_regions]
        #bbox_feat = bottom_up['norm_bb'][:self.num_regions]

        if self.split == 'train':
            return src, trg_inp, trg_gt
        else:
            return src, trg_inp, trg_gt, " ".join(entry[0]), ''.join(entry[1]), entry[-3], entry[-2], entry[-1]

    def __len__(self):
        return len(self.data)


def split(string):
    output = []
    buf_str = ""
    for s in string:
        if s == "(":
            string = buf_str.strip()
            if string:
                output.append(string)
            output.append("(")
            buf_str = ""
        elif s == ")":
            string = buf_str.strip()
            if string:
                output.append(string)
            output.append(")")
            buf_str = ""
        elif s == ",":
            string = buf_str.strip()
            if string:
                output.append(string)
            output.append(",")
            buf_str = ""
        else:
            buf_str += s
    return output


def generate_pairs(entry):
    if entry[2]:
        output = []
        for r in entry[2]:
            _, p = r.split('=')
            sub_p = split(p)
            output.extend(sub_p)
            output.append(";")
        del output[-1]
    else:
        output = []
    question = word_tokenize(entry[1])
    return (question, output, entry[0], entry[-2], entry[-1])


def create_pairs(filename, split):
    examples = []
    with open(filename) as f:
        data = json.load(f)
    print("total {} programs".format(len(data)))
    if split == 'submission':
        data = map(lambda x: (x[1]['imageId'], x[1]['question'], [], x[0], 'unknown'), data.items())

    cores = multiprocessing.cpu_count()
    print("using parallel computing with {} cores".format(cores))
    pool = Pool(cores)

    r = pool.map(generate_pairs, data)

    pool.close()
    pool.join()

    with open('{}/{}_pairs.json'.format(args.output, split), 'w') as f:
        json.dump(r, f)


def create_vocab(input_file):
    counter = Counter()

    with open(input_file) as f:
        r = json.load(f)

    for (q, p, _, _) in r:
        counter.update(q)
        counter.update(p)

    print('finished reading the question-program pairs')
    vocab = {"[PAD]": Constants.PAD, "[EOS]": Constants.EOS, "[UNK]": Constants.UNK, "[SOS]": Constants.SOS}
    for (w, freq) in counter.most_common():
        vocab[w] = len(vocab)

    with open('{}/full_vocab.json'.format(args.meta), 'w') as f:
        json.dump(vocab, f)


def get_batch(data, batch_size):
    examples = []
    length = len(data)
    intervals = (length // batch_size) + 1
    for i in range(intervals):
        yield data[i * batch_size: min(length, (i + 1) * batch_size)]


def train(option):
    with open('{}/full_vocab.json'.format(args.output), 'r') as f:
        vocab = json.load(f)
        inv_vocab = {v: k for k, v in vocab.items()}

    dataset = GQA(vocab, option)

    model = TransformerDecoder(len(vocab), args.hidden_dim, 3, args.hidden_dim, 4)
    print("loading the pre-trained word vectors from {}".format(args.word_glove))
    model = nn.DataParallel(model)
    model.to(device)

    if option == 'train':
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss(ignore_index=Constants.PAD)

        model.train()

        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
        epoch_loss = 0

        for epoch in range(args.max_epoch):
            for idx, batch in enumerate(data_loader):
                batch = tuple(Variable(t).to(device) for t in batch)
                src, trg_inp, trg_gt = batch

                model.zero_grad()
                optimizer.zero_grad()

                output = model(src_seq=src, tgt_seq=trg_inp)

                loss = criterion(output.view(-1, output.shape[-1]), trg_gt.view(-1))

                loss.backward()

                optimizer.step()

                if idx % 100 == 0:
                    print("step {}, loss = {}".format(idx, loss.item()))

                epoch_loss += loss.item()

            torch.save(model.state_dict(), 'models/seq2seq_ep{}.pt'.format(epoch))

    elif option in ['trainval_unbiased', 'testdev', 'submission']:
        model.load_state_dict(torch.load(args.load_from))
        print("loading the model from {}".format(args.load_from))

        model.eval()
        success, fail = 0, 0
        generated = []
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

        start_time = time.time()
        for idx, batch in enumerate(data_loader):
            orig_src, trg, imageId, questionId, answers = batch[3:]
            batch = tuple(Variable(t).to(device) for t in batch[:3])
            src, trg_inp, trg_gt = batch
            output = model.module.translate_batch(de_vocab=inv_vocab, src_seq=src, max_token_seq_len=args.max_len)
            for predicted, src, gt, im, qid, a in zip(output, orig_src, trg, imageId, questionId, answers):
                #sub_programs = list(filter(lambda x: len(x) > 0, (''.join(predicted)).split(';')))
                generated.append((im, src,  predicted.split(';'), qid, a))

                if gt:
                    if gt == predicted:
                        success += 1
                    elif 'choose' in gt and 'choose' in predicted:
                        success += 1
                    elif 'and' in gt or 'or' in gt:
                        success += 1
                    else:
                        if args.debug:
                            print(src, "@PRED:", predicted, "@GT:", gt)
                            print("#################################")
                        fail += 1

            # print "finish {}/{}".format(idx, len(data) // args.batch_size)
            if args.debug:
                if idx % 2 == 0 and idx > 0:
                    print("the success rate is {}/{}".format(success, fail + success))
                    break
            else:
                if idx % 10 == 0:
                    print("finished {}/{} with {} secs".format(idx, len(data_loader), time.time() - start_time))
                    start_time = time.time()

        print("the success rate is {}/{} = {}".format(success, fail + success, success / (fail + success + 0.)))
        # with open('questions/{}_programs_pred.json'.format(option), 'r') as f:
        #   prev_generated = json.load(f)
        if not args.debug:
            if option == 'submission':
                with open('questions/needed_submission_programs.json'.format(option), 'w') as f:
                    json.dump(generated, f, indent=2)
            elif option == 'testdev':
                with open('questions/testdev_pred{}_programs.json'.format(args.version), 'w') as f:
                    json.dump(generated, f, indent=2)
            else:
                with open('questions/{}_programs{}.json'.format(option, args.version), 'w') as f:
                    json.dump(generated, f, indent=2)
        else:
            print("Done with Debugging")


if args.do_preprocess:
    create_pairs('questions/trainval_all_programs.json', 'train')
    print("finished creating the pairs")
    create_vocab('{}/train_pairs.json'.format(args.output))
    create_pairs('questions/trainval_balanced_programs.json', 'train_balanced')
    create_pairs('questions/testdev_balanced_programs.json', 'testdev')
    create_pairs('questions/needed_submission_questions.json', 'submission')
elif args.do_train:
    train('train')
elif args.do_trainval_unbiased:
    train('trainval_unbiased')
elif args.do_testdev:
    train('testdev')
elif args.do_submission:
    train('submission')
else:
    print("unsupported")
