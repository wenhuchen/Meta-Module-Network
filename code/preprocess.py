import re
import nltk
import json
import sys
import json
import time
from nltk.stem import WordNetLemmatizer
import json
import sys
import Constants
import numpy as np
import random

lemmatizer = WordNetLemmatizer()


def add1(string, extra):
    #nums = string[1:-1].split(',')
    #nums = [str(int(_) + extra) for _ in nums]
    new_string = ""
    for c in string:
        if c.isdigit():
            new_string += str(int(c) + extra)
        else:
            new_string += c
    return new_string


def filter_field(string):
    output = re.search(r' ([^ ]+)\b', string).group()[2:]
    if 'not(' in output:
        return re.search(r'\(.+$', output).group()[1:], True
    else:
        return output, False


def filter_parenthesis(string):
    objects = re.search(r'\(.+\)', string).group()[1:-1]
    if objects == '-':
        return '[]'
    else:
        return '[{}]'.format(objects)


def filter_squre(string):
    indexes = re.search(r'\[.+\]', string).group()
    if ',' in indexes:
        return ','.join(['[{}]'.format(_.strip()) for _ in indexes[1:-1].split(',')])
    else:
        return indexes


def extract_rel(string):
    subject = re.search(r'^([^,]+),', string).group()[:-1]
    relation = re.search(r',(.+),', string).group()[1:-1]
    try:
        o_s = re.search(r',(o|s) ', string).group()[1:-1]
        if 's' in o_s:
            return subject, relation, True
        else:
            return subject, relation, False
    except:
        return subject, relation, None


def extract_query_key(string):
    if 'name' in string:
        return 'name'
    elif 'hposition' in string:
        return 'hposition'
    elif 'vposition' in string:
        return 'vposition'
    else:
        return 'attributes'


def split_rel(string):
    subject = re.search(r'([^,]+),', string).group()[:-1]
    relation1 = re.search(r',(.+)\|', string).group()[1:-1]
    relation2 = re.search(r'\|(.+),', string).group()[1:-1]
    o_s = re.search(r',(o|s)', string).group()[1:-1]
    if 's' in o_s:
        return subject, relation1, relation2, True
    else:
        return subject, relation1, relation2, False


def split_attr(string):
    attr1 = re.search(r'(.+)\|', string).group()[2:-1]
    attr2 = re.search(r'\|(.+) ', string).group()[1:-1]
    return attr1, attr2


def shuffle(string):
    attrs = string.split('|')
    random.shuffle(attrs)
    attr1, attr2 = attrs
    return attr1, attr2


def preprocess(raw_data, output_path, formal=False):
    symbolic_programs = []
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    keys = list(raw_data.keys())
    print("total {} programs".format(len(keys)))
    success, fail = 0, 0

    for idx in range(len(keys)):
        imageId = raw_data[keys[idx]]['imageId']
        question = raw_data[keys[idx]]['question']
        program = raw_data[keys[idx]]['semantic']
        answer = raw_data[keys[idx]]['answer']

        new_programs = []
        # try:
        for i, prog in enumerate(program):
            if prog['dependencies']:
                subject = ",".join(["[{}]".format(_) for _ in prog['dependencies']])

            if '(' in prog['argument'] and ')' in prog['argument'] and 'not(' not in prog['argument']:
                result = filter_parenthesis(prog['argument'])
            else:
                result = '?'

            if prog['operation'] == 'select':
                if prog['argument'] == 'scene':
                    # new_programs.append('{}=scene()'.format(result))
                    flag = 'full'
                else:
                    new_programs.append('{}=select({})'.format(
                        result, lemmatizer.lemmatize(prog['argument'].split(' ')[0])))
                    flag = 'partial'

            elif prog['operation'] == 'relate':
                # print prog['argument']
                name, relation, reverse = extract_rel(prog['argument'])
                if reverse == None:
                    new_programs.append('{}=relate_attr({}, {}, {})'.format(result, subject, relation, name))
                else:
                    if reverse:
                        if name != '_':
                            name = lemmatizer.lemmatize(name)
                            new_programs.append('{}=relate_inv_name({}, {}, {})'.format(
                                result, subject, relation, name))
                        else:
                            new_programs.append('{}=relate_inv({}, {})'.format(result, subject, relation))
                    else:
                        if name != '_':
                            name = lemmatizer.lemmatize(name)
                            new_programs.append('{}=relate_name({}, {}, {})'.format(result, subject, relation, name))
                        else:
                            new_programs.append('{}=relate({}, {})'.format(result, subject, relation))

            elif prog['operation'].startswith('query'):
                if prog['argument'] == "hposition":
                    new_programs.append('{}=query_h({})'.format(result, subject))
                elif prog['argument'] == "vposition":
                    new_programs.append('{}=query_v({})'.format(result, subject))

                elif prog['argument'] == "name":
                    new_programs.append('{}=query_n({})'.format(result, subject))
                else:
                    if flag == 'full':
                        new_programs.append('{}=query_f({})'.format(result, prog['argument']))
                    else:
                        new_programs.append('{}=query({}, {})'.format(result, subject, prog['argument']))

            elif prog['operation'] == 'exist':
                new_programs.append('{}=exist({})'.format(result, subject))

            elif prog['operation'] == 'or':
                new_programs.append('{}=or({})'.format(result, subject))

            elif prog['operation'] == 'and':
                new_programs.append('{}=and({})'.format(result, subject))

            elif prog['operation'].startswith('filter'):
                if prog['operation'] == 'filter hposition':
                    new_programs.append('{}=filter_h({}, {})'.format(result, subject, prog['argument']))

                elif prog['operation'] == 'filter vposition':
                    new_programs.append('{}=filter_h({}, {})'.format(result, subject, prog['argument']))

                else:
                    negative = 'not(' in prog['argument']
                    if negative:
                        new_programs.append('{}=filter_not({}, {})'.format(result, subject, prog['argument'][4:-1]))
                    else:
                        new_programs.append('{}=filter({}, {})'.format(result, subject, prog['argument']))

            elif prog['operation'].startswith('verify'):
                if prog['operation'] == 'verify':
                    new_programs.append('{}=verify({}, {})'.format(result, subject, prog['argument']))
                elif prog['operation'] == 'verify hposition':
                    new_programs.append('{}=verify_h({}, {})'.format(result, subject, prog['argument']))
                elif prog['operation'] == 'verify vposition':
                    new_programs.append('{}=verify_v({}, {})'.format(result, subject, prog['argument']))
                elif prog['operation'] == 'verify rel':
                    name, relation, reverse = extract_rel(prog['argument'])
                    name = lemmatizer.lemmatize(name)
                    if reverse:
                        new_programs.append('{}=verify_rel_inv({}, {}, {})'.format(result, subject, relation, name))
                    else:
                        new_programs.append('{}=verify_rel({}, {}, {})'.format(result, subject, relation, name))
                    # if reverse:
                    #    new_programs.append('?=relate_inv_name({}, {}, {})'.format(subject, relation, name))
                    #    new_programs.append('{}=exist([{}])'.format(result, len(new_programs) - 1))
                    # else:
                    #    new_programs.append('?=relate_name({}, {}, {})'.format(subject, relation, name))
                    #    new_programs.append('{}=exist([{}])'.format(result, len(new_programs) - 1))
                else:
                    if flag == 'full':
                        new_programs.append('{}=verify_f({})'.format(result, prog['argument']))
                    else:
                        new_programs.append('{}=verify({}, {})'.format(result, subject, prog['argument']))

            elif prog['operation'].startswith('choose'):
                if prog['operation'] == 'choose':
                    attr1, attr2 = shuffle(prog['argument'])
                    if flag == "full":
                        new_programs.append('{}=choose_f({}, {})'.format(result, attr1, attr2))
                    else:
                        new_programs.append('{}=choose({}, {}, {})'.format(result, subject, attr1, attr2))

                elif prog['operation'] == 'choose rel':
                    name, relation1, relation2, reverse = split_rel(prog['argument'])
                    relation1, relation2 = shuffle('{}|{}'.format(relation1, relation2))
                    name = lemmatizer.lemmatize(name)
                    if reverse:
                        new_programs.append('{}=choose_rel({}, {}, {}, {})'.format(
                            result, subject, name, relation1, relation2))
                    else:
                        new_programs.append('{}=choose_rel_inv({}, {}, {}, {})'.format(
                            result, subject, name, relation1, relation2))

                elif prog['operation'] == 'choose hposition':
                    attr1, attr2 = shuffle(prog['argument'])
                    new_programs.append('{}=choose_h({}, {}, {})'.format(result, subject, attr1, attr2))

                elif prog['operation'] == 'choose vposition':
                    attr1, attr2 = shuffle(prog['argument'])
                    new_programs.append('{}=choose_v({}, {}, {})'.format(result, subject, attr1, attr2))

                elif prog['operation'] == 'choose name':
                    attr1, attr2 = shuffle(prog['argument'])
                    attr1 = lemmatizer.lemmatize(attr1)
                    attr2 = lemmatizer.lemmatize(attr2)
                    new_programs.append('{}=choose_n({}, {}, {})'.format(result, subject, attr1, attr2))

                elif ' ' in prog['operation']:
                    attr = prog['operation'].split(' ')[1]
                    if len(prog['argument']) == 0:
                        new_programs.append('{}=choose_subj({}, {})'.format(result, subject, attr))
                    else:
                        attr1, attr2 = shuffle(prog['argument'])
                        if flag == "full":
                            new_programs.append('{}=choose_f({}, {})'.format(result, attr1, attr2))
                        else:
                            new_programs.append('{}=choose_attr({}, {}, {}, {})'.format(
                                result, subject, attr, attr1, attr2))

            elif prog['operation'].startswith('different'):
                if ' ' in prog['operation']:
                    attr = prog['operation'].split(' ')[1]
                    new_programs.append('{}=different_attr({}, {})'.format(result, subject, attr))
                else:
                    new_programs.append('{}=different({})'.format(result, subject))

            elif prog['operation'].startswith('same'):
                if ' ' in prog['operation']:
                    attr = prog['operation'].split(' ')[1]
                    new_programs.append('{}=same_attr({}, {})'.format(result, subject, attr))
                else:
                    new_programs.append('{}=same({})'.format(result, subject))

            elif prog['operation'] == 'common':
                new_programs.append('{}=common({})'.format(result, subject))

            else:
                raise ValueError("Unseen Function {}".format(prog))
            # if answer == "yes":
            #    answer = True
            # elif answer == "no":
            #    answer = False
            # elif 'choose' in new_programs[-1]:
            #    _, _, arguments = parse_program(new_programs[-1])
            #    if answer not in arguments:
            #        import pdb
            #        pdb.set_trace()
            # elif answer == "right" and 'choose' in new_programs[-1]:
            #    answer = 'to the right of'
            # elif answer == "left" and 'choose' in new_programs[-1]:
            #    answer = 'to the left of'

        symbolic_programs.append((imageId, question, new_programs, keys[idx], answer))
        success += 1

        # except Exception:
        #    print(program)
        #    fail += 1

        if idx % 10000 == 0:
            sys.stdout.write("finished {}/{} \r".format(success, fail))

    print("finished {}/{}".format(success, fail))
    with open(output_path, 'w') as f:
        json.dump(symbolic_programs, f, indent=2)


def create_inputs(splits, output):
    def find_all_nums(strings):
        nums = []
        for s in strings:
            if '[' in s and ']' in s:
                nums.append(int(s[1:-1]))
        return nums

    results = []
    for split in splits:
        # if split == 'submission':
        #    with open('questions/{}_programs_pred.json'.format(split)) as f:
        #        data = json.load(f)
        # else:
        with open('questions/{}_programs.json'.format(split)) as f:
            data = json.load(f)
            print("loading {}".format('questions/{}_programs.json'.format(split)))

        count = 0
        for idx, entry in enumerate(data):
            # for prog in entry[2]:
            programs = entry[2]
            rounds = []
            depth = {}
            cur_depth = 0
            tmp = []
            connection = []
            inputs = []
            returns = []
            tmp_connection = []
            for i, program in enumerate(programs):
                if isinstance(program, list):
                    _, func, args = Constants.parse_program(program[1])
                    returns.append(program[0])
                else:
                    _, func, args = Constants.parse_program(program)
                try:
                    if func == 'relate' or func == 'relate_inv':
                        inputs.append([func, None, None, args[1], None, None, None, None])
                    elif func == 'relate_attr':
                        inputs.append([func, None, None, args[1], args[2], None, None, None])
                    elif func == 'relate_name' or func == 'relate_inv_name':
                        inputs.append([func, None, None, args[1], args[2], None, None, None])
                    elif func == 'select':
                        inputs.append([func, None, None, None, args[0], None, None, None])
                    elif func == 'filter' or func == 'filter_not':
                        inputs.append([func, None, args[1], None, None, None, None, None])
                    elif func == 'filter_h' or func == 'filter_v':
                        inputs.append([func, None, None, None, None, args[1], None, None])
                    elif func == 'verify_h' or func == 'verify_v':
                        inputs.append([func, None, None, None, None, args[0], None, None])
                    elif func == 'query_n':
                        inputs.append([func, None, None, None, None, None, None, None])
                    elif func == 'query_h' or func == 'query_v':
                        inputs.append([func, None, None, None, None, None, None, None])
                    elif func == 'query':
                        inputs.append([func, args[1], None, None, None, None, None, None])
                    elif func == 'query_f':
                        inputs.append([func, args[0], None, None, None, None, None, None])
                    elif func == 'verify':
                        inputs.append([func, None, args[1], None, None, None, None, None])
                    elif func == 'verify_f':
                        inputs.append([func, None, args[0], None, None, None, None, None])
                    elif func == 'verify_rel' or func == 'verify_rel_inv':
                        inputs.append([func, None, None, args[1], args[2], None, None, None])
                    elif func in ['choose_n', 'choose_h', 'choose_v']:
                        inputs.append([func, None, None, None, None, None, args[1], args[2]])
                    elif func == 'choose':
                        inputs.append([func, None, None, None, None, None, args[1], args[2]])
                    elif func == 'choose_subj':
                        inputs.append([func, None, args[2], None, None, None, None, None])
                    elif func == 'choose_attr':
                        inputs.append([func, args[1], None, None, None, None, args[2], args[3]])
                    elif func == 'choose_f':
                        inputs.append([func, None, None, None, None, None, args[0], args[1]])
                    elif func == 'choose_rel_inv':
                        inputs.append([func, None, None, None, args[1], None, args[2], args[3]])
                    elif func in ['same_attr', 'different_attr']:
                        inputs.append([func, None, args[2], None, None, None, None, None])
                    elif func in ['exist', 'or', 'and', 'different', 'same', 'common']:
                        inputs.append([func, None, None, None, None, None, None, None])
                    else:
                        raise ValueError('unknown function {}'.format(func))
                except Exception:
                    print(program)
                    inputs.append([func, None, None, None, None, None, None, None])

                assert len(inputs[-1]) == 8
                if len(find_all_nums(args)) == 0:
                    tmp.append(program)
                    depth[i] = cur_depth
                    tmp_connection.append([i, i])

            connection.append(tmp_connection)
            cur_depth += 1
            rounds.append(tmp)

            while len(depth) < len(programs):
                tmp = []
                tmp_depth = {}
                tmp_connection = []
                for i, program in enumerate(programs):
                    if i in depth:
                        continue
                    else:
                        if isinstance(program, list):
                            _, func, args = Constants.parse_program(program[1])
                        else:
                            _, func, args = Constants.parse_program(program)
                        reliance = find_all_nums(args)
                        if all([_ in depth for _ in reliance]):
                            tmp.append(program)
                            tmp_depth[i] = cur_depth
                            for r in reliance:
                                if r > i:
                                    r = i - 1
                                tmp_connection.append([i, r])
                        else:
                            continue

                if len(tmp_depth) == 0 and len(tmp) == 0 and len(tmp_connection) == 0:
                    break
                else:
                    connection.append(tmp_connection)
                    rounds.append(tmp)
                    cur_depth += 1
                    depth.update(tmp_depth)

            results.append([entry[0], entry[1], returns, inputs, connection, entry[-2], entry[-1]])
            sys.stdout.write("finished {}/{} \r".format(idx, len(data)))

    with open(output, 'w') as f:
        json.dump(results, f, indent=2)


arg = sys.argv[1]
if arg == 'trainval_all':
    raw_data = {}
    start_time = time.time()
    with open('../gqa-questions/train_all_questions/train_all_questions_0.json') as f:
        raw_data.update(json.load(f))
    with open('../gqa-questions/train_all_questions/train_all_questions_1.json') as f:
        raw_data.update(json.load(f))
    with open('../gqa-questions/train_all_questions/train_all_questions_2.json') as f:
        raw_data.update(json.load(f))
    with open('../gqa-questions/train_all_questions/train_all_questions_3.json') as f:
        raw_data.update(json.load(f))
    with open('../gqa-questions/train_all_questions/train_all_questions_4.json') as f:
        raw_data.update(json.load(f))
    with open('../gqa-questions/train_all_questions/train_all_questions_5.json') as f:
        raw_data.update(json.load(f))
    with open('../gqa-questions/train_all_questions/train_all_questions_6.json') as f:
        raw_data.update(json.load(f))
    with open('../gqa-questions/train_all_questions/train_all_questions_7.json') as f:
        raw_data.update(json.load(f))
    with open('../gqa-questions/train_all_questions/train_all_questions_8.json') as f:
        raw_data.update(json.load(f))
    with open('../gqa-questions/train_all_questions/train_all_questions_9.json') as f:
        raw_data.update(json.load(f))
    with open('../gqa-questions/val_all_questions.json') as f:
        raw_data.update(json.load(f))
    preprocess(raw_data, 'questions/trainval_all_programs.json')

elif arg == 'create_balanced_programs':
    with open('questions/original/train_balanced_questions.json') as f:
        raw_data = json.load(f)
    with open('questions/original/val_balanced_questions.json') as f:
        raw_data.update(json.load(f))
    preprocess(raw_data, 'questions/trainval_balanced_programs.json')
    with open('questions/original/testdev_balanced_questions.json') as f:
        raw_dev_data = json.load(f)
    preprocess(raw_dev_data, 'questions/original/testdev_balanced_programs.json')

elif arg == 'create_all_inputs':
    create_inputs(['trainval_all_fully'], 'questions/trainval_all_fully_inputs.json')

elif arg == 'create_calibrated_inputs':
    create_inputs(['trainval_calibrated_fully'], 'questions/trainval_calibrated_fully_inputs.json')

elif arg == 'create_inputs':
    #create_inputs(['trainval_balanced'], 'questions/trainval_balanced_inputs.json')
    create_inputs(['testdev_balanced'], 'questions/testdev_balanced_inputs.json')
    create_inputs(['trainval_fully'], 'questions/trainval_fully_inputs.json')

elif arg == 'create_pred_inputs':
    create_inputs(['trainval_unbiased_fully'], 'questions/trainval_unbiased_fully_inputs.json')
    create_inputs(['testdev_pred'], 'questions/testdev_pred_inputs.json')

elif arg == 'submission_inputs':
    create_inputs(['needed_submission', 'overlapped_submission'], 'questions/submission_inputs.json')

elif arg == 'debug':
    with open('questions/needed_submission_programs.json') as f:
        data = json.load(f)
    for entry in data:
        for sub_program in entry[2]:
            try:
                Constants.parse_program(sub_program)
            except Exception:
                print(sub_program)

elif arg == 'glove_emb':
    def loadGloveModel(gloveFile):
        print("Loading Glove Model")
        f = open(gloveFile, 'r')
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print("Done.", len(model), " words loaded!")
        return model

    emb = loadGloveModel('glove/glove.6B.300d.txt')

    def save(inputs, outputs):
        with open(inputs) as f:
            vocab = json.load(f)

        found, miss = 0, 0
        en_emb = np.zeros((len(vocab), 300), 'float32')
        for w, i in vocab.items():
            if w.lower() in emb:
                en_emb[i] = emb[w.lower()]
                found += 1
            elif ' ' in w:
                for w_elem in w.split(' '):
                    if w_elem.lower() in emb:
                        en_emb[i] += emb[w_elem.lower()]
            else:
                print(w)
                miss += 1

        print("found = {}, miss = {}".format(found, miss))
        np.save(outputs, en_emb)

    save('meta_info/full_vocab.json', 'models/en_emb.npy')
    #save('nl2prog/concept_vocab.json', 'models/concept_emb.npy')
else:
    raise NotImplementedError
