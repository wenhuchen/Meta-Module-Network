from nltk.tokenize import word_tokenize
import json
import h5py
import Constants
import os
import numpy as np
from torch.utils.data import Dataset
import pickle
import torch
from torch import nn


class GQA(Dataset):
    def __init__(self, **args):
        self.mode = args['mode']
        self.split = args['split']
        if args['forbidden'] != '':
            with open(args['forbidden'], 'r') as f:
                self.forbidden = json.load(f)
            self.forbidden = set(self.forbidden)
        else:
            self.forbidden = set([])
        
        with open('questions/{}_inputs.json'.format(self.split), 'r') as f:
            self.data = json.load(f)
        print("loading data from {}".format(
            'questions/{}_inputs.json'.format(self.split)))

        if self.split == 'trainval_all_fully':
            with open('questions/trainval_calibrated_fully_inputs.json') as f:
                self.data += json.load(f)
            print("loading additional data from questions/trainval_calibrated_fully_inputs.json")

        with open(args['object_info']) as f:
            self.object_info = json.load(f)

        database = set(self.object_info.keys())
        self.data = list(filter(lambda x: x[0] in database, self.data))
        print("there are in total {} instances before validation removal".format(len(self.data)))

        self.data = list(filter(lambda x: x[-2] not in self.forbidden, self.data))
        print("there are in total {} instances".format(len(self.data)))

        self.vocab = args['vocab']
        self.answer_vocab = args['answer']
        self.num_tokens = args['num_tokens']
        self.num_regions = args['num_regions']
        self.LENGTH = args['length']
        self.MAX_LAYER = args['max_layer']

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)


class GQA_v1(GQA):
    def __init__(self, **args):
        super(GQA_v1, self).__init__(**args)
        self.object_file = args['object_file']

    def __getitem__(self, index):
        entry = self.data[index]
        image_id = entry[0]
        question = entry[1]
        inputs = entry[3]
        connection = entry[4]
        obj_info = self.object_info[image_id]
        # spatial_info = self.spatial_info[image_id]

        length = min(len(inputs), self.LENGTH)
        # layer = min(len(connection), self.MAX_LAYER)

        # Prepare Question
        idxs = word_tokenize(question)[:self.num_tokens]
        question = [self.vocab.get(_, Constants.UNK) for _ in idxs]
        question += [Constants.PAD] * (self.num_tokens - len(idxs))
        question = np.array(question, 'int64')

        question_masks = np.zeros((len(question), ), 'float32')
        question_masks[:len(idxs)] = 1.
        # Prepare Program
        program = np.zeros((self.LENGTH, 8), 'int64')
        for i in range(length):
            for j, text in enumerate(inputs[i]):
                if text is not None:
                    program[i][j] = self.p_vocab.get(text, Constants.UNK)

        # Prepare Program mask
        program_masks = np.zeros((self.LENGTH, ), 'float32')
        program_masks[:length] = 1.

        # Prepare Program Transition Mask
        transition_masks = np.zeros(
            (self.MAX_LAYER, self.LENGTH, self.LENGTH), 'uint8')
        depth = np.zeros((self.LENGTH, ), 'int64')
        for i in range(self.MAX_LAYER):
            for j in range(self.LENGTH):
                transition_masks[i][j][j] = 1
            if i < len(connection):
                for idx, idy in connection[i]:
                    transition_masks[i][idx][idy] = 1
                    depth[idx] = i
        depth[length - 1] = self.MAX_LAYER - 1

        # Prepare Vision Feature
        with h5py.File(self.object_file, 'r') as db:
            vis_mask = np.zeros((self.num_regions, ), 'float32')
            vis_mask[:obj_info['objectsNum']] = 1.
            object_feat = db['features'][obj_info['index']][:self.num_regions]
            span = object_feat.shape[0]
            coordinates = db['bboxes'][obj_info['index']][:span]

        if self.mode == 'train':
            returns = entry[2]
            intermediate_idx = np.full((self.LENGTH, ), -1, 'int64')
            intersect_iou = np.full((length - 1, span), 0., 'float32')
            for idx in range(length - 1):
                if isinstance(returns[idx], list):
                    if returns[idx] == [-1, -1, -1, -1]:
                        intermediate_idx[idx] = self.num_regions
                    else:
                        gt_coordinate = (returns[idx][0], returns[idx][1],
                                         returns[idx][0] + returns[idx][2],
                                         returns[idx][1] + returns[idx][3])
                        # contained_iou = np.full((len(coordinates), ), 0., 'float32')
                        for i in range(min(obj_info['objectsNum'], span)):
                            intersect, contain = Constants.intersect(
                                gt_coordinate, coordinates[i], True, 'x1y1x2y2')
                            intersect_iou[idx][i] = intersect + 0.2 * contain

                        if max(intersect_iou[idx]) > 0.5:
                            intermediate_idx[idx] = np.argmax(
                                intersect_iou[idx])
                elif returns[idx] == True:
                    intermediate_idx[idx] = self.num_regions + 1
                elif returns[idx] == False:
                    intermediate_idx[idx] = self.num_regions + 2
                else:
                    pass
        else:
            intermediate_idx = 0

        coordinates[:, 0] = coordinates[:, 0] / obj_info['width']
        coordinates[:, 2] = coordinates[:, 2] / obj_info['width']
        coordinates[:, 1] = coordinates[:, 1] / obj_info['height']
        coordinates[:, 3] = coordinates[:, 3] / obj_info['height']
        x1 = (coordinates[:, 2] - coordinates[:, 0]).reshape((span, 1))
        y1 = (coordinates[:, 3] - coordinates[:, 1]).reshape((span, 1))

        box_feat = np.concatenate([coordinates, x1, y1], -1)

        # Prepare index selection
        index = length - 1

        # Prepare answer
        if self.split == 'submission':
            answer_id = entry[-1]
        else:
            answer_id = self.answer_vocab.get(entry[-1], Constants.UNK)

        return question, question_masks, program, program_masks, transition_masks, object_feat, \
            box_feat, vis_mask, index, depth, intermediate_idx, answer_id

    def __len__(self):
        return len(self.data)


class GQA_v2(GQA):
    def __init__(self, **args):
        super(GQA_v2, self).__init__(**args)
        self.folder = args['folder']
        self.threshold = args['threshold']
        self.contained_weight = args['contained_weight']
        self.cutoff = args['cutoff']
        self.distribution = args['distribution']

    def __getitem__(self, index):
        entry = self.data[index]
        obj_info = self.object_info[entry[0]]
        if not entry[0].startswith('n'):
            if len(entry[0]) < 7:
                entry[0] = "0" * (7 - len(entry[0])) + entry[0]

        image_id = entry[0]
        question = entry[1]
        inputs = entry[3]
        connection = entry[4]
        questionId = entry[-2]

        length = min(len(inputs), self.LENGTH)

        # Prepare Question
        idxs = word_tokenize(question)[:self.num_tokens]
        question = [self.vocab.get(_, Constants.UNK) for _ in idxs]
        question += [Constants.PAD] * (self.num_tokens - len(idxs))
        question = np.array(question, 'int64')

        question_masks = np.zeros((len(question), ), 'float32')
        question_masks[:len(idxs)] = 1.
        # Prepare Program
        program = np.zeros((self.LENGTH, 8), 'int64')
        depth = np.zeros((self.LENGTH, ), 'int64')
        for i in range(length):
            for j, text in enumerate(inputs[i]):
                if text is not None:
                    program[i][j] = self.vocab.get(text, Constants.UNK)

        # Prepare Program mask
        program_masks = np.zeros((self.LENGTH, ), 'float32')
        program_masks[:length] = 1.

        # Prepare Program Transition Mask
        transition_masks = np.zeros(
            (self.MAX_LAYER, self.LENGTH, self.LENGTH), 'uint8')
        activate_mask = np.zeros((self.MAX_LAYER, self.LENGTH), 'float32')
        for i in range(self.MAX_LAYER):
            if i < len(connection):
                for idx, idy in connection[i]:
                    transition_masks[i][idx][idy] = 1
                    depth[idx] = i
                    activate_mask[i][idx] = 1
            for j in range(self.LENGTH):
                if activate_mask[i][j] == 0:
                    # As a placeholder
                    transition_masks[i][j][j] = 1
                else:
                    pass

        vis_mask = np.zeros((self.num_regions, ), 'float32')
        # Prepare Vision Feature
        bottom_up = np.load(os.path.join(
            self.folder, 'gqa_{}.npz'.format(image_id)))
        adaptive_num_regions = min(
            (bottom_up['conf'] > self.threshold).sum(), self.num_regions)

        # Cut off the bottom up features
        object_feat = bottom_up['features'][:adaptive_num_regions]
        bbox_feat = bottom_up['norm_bb'][:adaptive_num_regions]
        vis_mask[:bbox_feat.shape[0]] = 1.
        # Padding zero
        if object_feat.shape[0] < self.num_regions:
            padding = self.num_regions - object_feat.shape[0]
            object_feat = np.concatenate([object_feat, np.zeros(
                (padding, object_feat.shape[1]), 'float32')], 0)
        if bbox_feat.shape[0] < self.num_regions:
            padding = self.num_regions - bbox_feat.shape[0]
            bbox_feat = np.concatenate([bbox_feat, np.zeros(
                (padding, bbox_feat.shape[1]), 'float32')], 0)
        num_regions = bbox_feat.shape[0]

        # exist = np.full((self.LENGTH, ), -1, 'float32')
        if self.mode == 'train':
            returns = entry[2]
            intermediate_idx = np.full(
                (self.LENGTH, num_regions + 1), 0, 'float32')
            intersect_iou = np.full(
                (length - 1, num_regions + 1), 0., 'float32')
            for idx in range(length - 1):
                if isinstance(returns[idx], list):
                    if returns[idx] == [-1, -1, -1, -1]:
                        intermediate_idx[idx][num_regions] = 1
                    else:
                        gt_coordinate = (returns[idx][0] / (obj_info['width'] + 0.),
                                         returns[idx][1] / (obj_info['height'] + 0.),
                                         (returns[idx][2] + returns[idx][0]) / (obj_info['width'] + 0.),
                                         (returns[idx][3] + returns[idx][1]) / (obj_info['height'] + 0.))
                        for i in range(num_regions):
                            intersect, contain = Constants.intersect(
                                gt_coordinate, bbox_feat[i, :4], True, 'x1y1x2y2')
                            intersect_iou[idx][i] = intersect  # + self.contained_weight * contain

                        # if self.distribution:
                            #mask = (intersect_iou[idx] > self.cutoff).astype('float32')
                            #intersect_iou[idx] *= mask
                        intermediate_idx[idx] = intersect_iou[idx] / (intersect_iou[idx].sum() + 0.001)
                        # else:
                        #    intermediate_idx[idx] = (intersect_iou[idx] > self.cutoff).astype('float32')
                        #    intermediate_idx[idx] = intermediate_idx[idx] / (intermediate_idx[idx].sum() + 0.001)

        else:
            intermediate_idx = 0

        # Prepare index selection
        index = length - 1
        # Prepare answer
        answer_id = self.answer_vocab.get(entry[-1], Constants.UNK)
        return question, question_masks, program, program_masks, transition_masks, activate_mask, object_feat, \
            bbox_feat, vis_mask, index, depth, intermediate_idx, answer_id, questionId


class GQA_v3(GQA):
    def __init__(self, **args):
        super(GQA_v3, self).__init__(**args)
        self.folder = args['folder']
        self.threshold = args['threshold']
        self.contained_weight = args['contained_weight']
        self.cutoff = args['cutoff']
        self.distribution = args['distribution']

    def __getitem__(self, index):
        entry = self.data[index]
        obj_info = self.object_info[entry[0]]
        # spatial_info = self.spatial_info[entry[0]]
        if not entry[0].startswith('n'):
            if len(entry[0]) < 7:
                entry[0] = "0" * (7 - len(entry[0])) + entry[0]

        image_id = entry[0]
        question = entry[1]
        inputs = entry[3]
        connection = entry[4]
        questionId = entry[-2]

        length = min(len(inputs), self.LENGTH)

        # Prepare Question
        idxs = word_tokenize(question)[:self.num_tokens]
        question = [self.vocab.get(_, Constants.UNK) for _ in idxs]
        question += [Constants.PAD] * (self.num_tokens - len(idxs))
        question = np.array(question, 'int64')

        question_masks = np.zeros((len(question), ), 'float32')
        question_masks[:len(idxs)] = 1.
        # Prepare Program
        program = np.zeros((self.LENGTH, 8), 'int64')
        depth = np.zeros((self.LENGTH, ), 'int64')
        switch = np.zeros((self.LENGTH, 3), 'float32')

        for i in range(length):
            if inputs[i][0] in Constants.OBJECT_FUNCS:
                switch[i][0] = 1.
            elif inputs[i][0] in Constants.BINARY_FUNCS:
                switch[i][1] = 1.
            elif inputs[i][0] in Constants.STRING_FUNCS:
                switch[i][2] = 1.
            else:
                raise NotImplementedError

            for j, text in enumerate(inputs[i]):
                if text is not None:
                    program[i][j] = self.vocab.get(text, Constants.UNK)

        # Prepare Program mask
        program_masks = np.zeros((self.LENGTH, ), 'float32')
        program_masks[:length] = 1.

        # Prepare Program Transition Mask
        transition_masks = np.zeros(
            (self.MAX_LAYER, self.LENGTH, self.LENGTH), 'uint8')
        activate_mask = np.zeros((self.MAX_LAYER, self.LENGTH), 'float32')
        for i in range(self.MAX_LAYER):
            if i < len(connection):
                for idx, idy in connection[i]:
                    transition_masks[i][idx][idy] = 1
                    depth[idx] = i
                    activate_mask[i][idx] = 1
            for j in range(self.LENGTH):
                if activate_mask[i][j] == 0:
                    # As a placeholder
                    transition_masks[i][j][j] = 1
                else:
                    pass

        vis_mask = np.zeros((self.num_regions, ), 'float32')
        # Prepare Vision Feature
        bottom_up = np.load(os.path.join(
            self.folder, 'gqa_{}.npz'.format(image_id)))
        adaptive_num_regions = min(
            (bottom_up['conf'] > self.threshold).sum(), self.num_regions)

        # Cut off the bottom up features
        object_feat = bottom_up['features'][:adaptive_num_regions]
        bbox_feat = bottom_up['norm_bb'][:adaptive_num_regions]
        vis_mask[:bbox_feat.shape[0]] = 1.
        # Padding zero
        if object_feat.shape[0] < self.num_regions:
            padding = self.num_regions - object_feat.shape[0]
            object_feat = np.concatenate([object_feat, np.zeros(
                (padding, object_feat.shape[1]), 'float32')], 0)
        if bbox_feat.shape[0] < self.num_regions:
            padding = self.num_regions - bbox_feat.shape[0]
            bbox_feat = np.concatenate([bbox_feat, np.zeros(
                (padding, bbox_feat.shape[1]), 'float32')], 0)
        num_regions = bbox_feat.shape[0]

        # exist = np.full((self.LENGTH, ), -1, 'float32')
        if self.mode == 'train':
            returns = entry[2]
            intermediate_idx = np.full(
                (self.LENGTH, num_regions + 1), 0, 'float32')
            intersect_iou = np.full(
                (length - 1, num_regions + 1), 0., 'float32')
            for idx in range(length - 1):
                if isinstance(returns[idx], list):
                    if returns[idx] == [-1, -1, -1, -1]:
                        intermediate_idx[idx][num_regions] = 1
                    else:
                        gt_coordinate = (returns[idx][0] / (obj_info['width'] + 0.),
                                         returns[idx][1] / (obj_info['height'] + 0.),
                                         (returns[idx][2] + returns[idx][0]) / (obj_info['width'] + 0.),
                                         (returns[idx][3] + returns[idx][1]) / (obj_info['height'] + 0.))
                        for i in range(num_regions):
                            intersect, contain = Constants.intersect(
                                gt_coordinate, bbox_feat[i, :4], True, 'x1y1x2y2')
                            intersect_iou[idx][i] = intersect  # + self.contained_weight * contain

                        intermediate_idx[idx] = intersect_iou[idx] / (intersect_iou[idx].sum() + 0.001)

        else:
            intermediate_idx = 0

        # Prepare index selection
        index = length - 1
        # Prepare answer
        answer_id = self.answer_vocab.get(entry[-1], Constants.UNK)
        return question, question_masks, program, program_masks, transition_masks, activate_mask, object_feat, \
            bbox_feat, vis_mask, index, depth, switch, intermediate_idx, answer_id, questionId


class BCEWithMask(nn.Module):
    def __init__(self, ignore_index):
        super(BCEWithMask, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, gt):
        mask = (gt != self.ignore_index).float()
        prob = torch.sigmoid(logits)
        loss = -torch.log(prob) * gt + (-torch.log(1 - prob)) * (1 - gt)
        length = torch.sum(mask)
        loss = torch.sum(loss * mask) / length
        return loss


class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, prob, logits):
        length = (prob.sum(-1) > 0.001).sum()
        pred_prob = torch.softmax(logits, -1)
        loss = -prob * torch.log(pred_prob)
        loss = torch.sum(loss, -1)
        loss = torch.sum(loss) / length
        return loss
