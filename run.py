import json
from nltk.stem import WordNetLemmatizer
from multiprocessing import Pool
import multiprocessing
import argparse
import Constants
import random
from collections import Counter
from API_provider import APIS
import pdb


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_trainval', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_trainval_all', default=False, action="store_true",
                        help="whether to train or test the model")
    parser.add_argument('--do_trainval_calibrated', default=False, action="store_true",
                        help="whether to train or test the model")
    parser.add_argument('--do_trainval_unbiased', default=False, action="store_true",
                        help="whether to train or test the model")
    parser.add_argument('--do_testdev_balanced', default=False, action="store_true",
                        help="whether to train or test the model")
    parser.add_argument('--do_testdev_balanced_pred', default=False,
                        action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_submission', default=False, action="store_true",
                        help="whether to train or test the model")
    parser.add_argument('--do_construct', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_test', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--debug', default=False, action="store_true", help="whether to train or test the model")
    args = parser.parse_args()
    return args


def parse_program(string):
    if '=' in string:
        result, function = string.split('=')
    else:
        function = string
        result = "?"

    func, arguments = function.split('(')
    if len(arguments) == 1:
        return result, func, []
    else:
        arguments = list(map(lambda x: x.strip(), arguments[:-1].split(',')))
        return result, func, arguments


def execute(entry):
    imageId = entry[0]
    graph = scene_graph[imageId]
    questionId = entry[-2]
    answer = entry[-1]
    program = entry[2]
    execution_buffer = []
    bounding_box = []
    results = []
    for i, sub_program in enumerate(program):
        if len(sub_program) == 0:
            continue

        result, func, arguments = parse_program(sub_program)
        results.append(result)

        if len(arguments) > 1 and (func == 'same' or func == 'different'):
            func = 'same_v2' if func == 'same' else 'different_v2'

        inputs = [graph]

        for arg in arguments:
            if '[' in arg and ']' in arg:
                coreference = int(arg[1:-1])
                # if coreference >= len(execution_buffer):
                #   coreference = -1
                inputs.append(execution_buffer[coreference])
            else:
                inputs.append(arg)
        try:
            returns = APIS[func](*inputs)
            if func.startswith('choose_rel'):
                if returns == 'to the left of':
                    returns = 'left'
                elif returns == 'to the right of':
                    returns = 'right'
                elif returns == 'in front of':
                    returns = 'front'
                elif returns in ['above', 'under', 'behind']:
                    pass
        except Exception:
            if args.debug:
                print("EXCEPTION", "@", entry[1], "@", imageId, "@", program, "@", answer)

            if args.do_submission:
                return {'questionId': answer, 'prediction': None}
            elif args.do_construct:
                returns = None
            else:
                return -1

        if i == len(program) - 1:
            bounding_box.append(None)
            if isinstance(returns, tuple):
                execution_buffer.append(returns[0])
            else:
                execution_buffer.append(returns)
        else:
            if isinstance(returns, list):
                if result[0] == '[' and result[-1] == ']':
                    if result == '[]':
                        bounding_box.append((-1, -1, -1, -1))
                    else:
                        result = result[1:-1].split(',')[0]
                        bounding_box.append((graph[result]['x'], graph[result]['y'],
                                             graph[result]['w'], graph[result]['h']))
                else:
                    if returns:
                        bounding_box.append((graph[returns[0]]['x'], graph[returns[0]]['y'],
                                             graph[returns[0]]['w'], graph[returns[0]]['h']))
                    else:
                        bounding_box.append((-1, -1, -1, -1))
                execution_buffer.append(returns)
            elif isinstance(returns, tuple):
                if returns[0]:
                    bounding_box.append((graph[returns[1]]['x'], graph[returns[1]]['y'],
                                         graph[returns[1]]['w'], graph[returns[1]]['h']))
                else:
                    bounding_box.append((-1, -1, -1, -1))
                execution_buffer.append(returns[0])
            else:
                bounding_box.append((-1, -1, -1, -1))
                #print(program, sub_program, returns)
                execution_buffer.append(returns)
                #raise NotImplementedError

    if execution_buffer[-1] == False:
        execution_buffer[-1] = 'no'
    elif execution_buffer[-1] == True:
        execution_buffer[-1] = 'yes'

    for i in range(len(program)):
        if '=' in program[i]:
            funcs = program[i].split('=')[1]
        else:
            funcs = program[i]

        program[i] = (bounding_box[i], funcs)

    if args.do_submission:
        if isinstance(execution_buffer[-1], list):
            execution_buffer[-1] = None
        return {'questionId': answer, 'prediction': execution_buffer[-1]}
    elif args.do_construct:
        entry[2] = program
        return entry
    else:
        if execution_buffer[-1] != answer:
            if args.debug:
                print("ERROR", "@", entry[1], "@", entry[-2], "@", imageId,
                      "@", program, "@", execution_buffer[-1], "@", answer)

            if execution_buffer[-1] is None:
                return -1
            else:
                return 0
        else:
            return 1


if __name__ == "__main__":
    args = parse_opt()
    if args.do_trainval_all:
        with open('questions/trainval_all_programs.json') as f:
            data = json.load(f)
        with open('sceneGraphs/trainval_bounding_box.json') as f:
            scene_graph = json.load(f)
    elif args.do_trainval:
        with open('questions/trainval_balanced_programs.json') as f:
            data = json.load(f)
        with open('sceneGraphs/trainval_bounding_box.json') as f:
            scene_graph = json.load(f)
        print("loading questions/trainval_balanced_programs.json")
    elif args.do_trainval_calibrated:
        with open('questions/trainval_calibrated_programs.json') as f:
            data = json.load(f)
        with open('sceneGraphs/trainval_bounding_box.json') as f:
            scene_graph = json.load(f)
        print("loading questions/trainval_calibrated_programs.json")
    elif args.do_trainval_unbiased:
        with open('questions/trainval_unbiased_programs.json') as f:
            data = json.load(f)
        with open('sceneGraphs/trainval_bounding_box.json') as f:
            scene_graph = json.load(f)
        print("loading questions/trainval_unbiased_programs.json")
    elif args.do_testdev_balanced:
        with open('questions/testdev_balanced_programs.json') as f:
            data = json.load(f)
        with open('sceneGraphs/testdev_bounding_box_pred.json') as f:
            scene_graph = json.load(f)
    elif args.do_testdev_balanced_pred:
        with open('questions/testdev_pred_programs.json') as f:
            data = json.load(f)
        with open('sceneGraphs/testdev_bounding_box_pred.json') as f:
            scene_graph = json.load(f)
    elif args.do_submission:
        with open('questions/submission_programs_pred.json') as f:
            data = json.load(f)
        with open('sceneGraphs/submission_bounding_box_pred.json') as f:
            scene_graph = json.load(f)
    else:
        print("continue with other functionality")
        pass
        #raise ValueError("Not Yet Implemented")

    if args.do_construct:
        cores = multiprocessing.cpu_count()
        print("using parallel computing with {} cores".format(cores))
        pool = Pool(cores)

        r1 = pool.map(execute, data)

        pool.close()
        pool.join()
        """
        with open('questions/trainval_fully_programs.json', 'r') as f:
            r = json.load(f)

        r = list(filter(lambda x: x is not None, r))
        with open('questions/trainval_fully_programs.json', 'w') as f:
            json.dump(r + r1, f, indent=2)
        """
        if args.do_trainval:
            with open('questions/trainval_fully_programs.json', 'w') as f:
                json.dump(r1, f, indent=2)
        elif args.do_trainval_unbiased:
            with open('questions/trainval_unbiased_fully_programs.json', 'w') as f:
                json.dump(r1, f, indent=2)
        elif args.do_trainval_all:
            with open('questions/trainval_all_fully_programs.json', 'w') as f:
                json.dump(r1, f, indent=2)
        elif args.do_trainval_calibrated:
            with open('questions/trainval_calibrated_fully_programs.json', 'w') as f:
                json.dump(r1, f, indent=2)
        else:
            raise NotImplementedError
    else:
        cores = multiprocessing.cpu_count()
        print("using parallel computing with {} cores".format(cores))
        pool = Pool(cores)

        if args.debug:
            # random.shuffle(data)
            r = []
            for entry in data[:100]:
                r.append(execute(entry))
        else:
            r = pool.map(execute, data)

        pool.close()
        pool.join()

        if args.do_submission:
            counter = Counter()
            counter.update([_['prediction'] for _ in r])
            print(counter.most_common())
            with open('sceneGraphs/submission_results.json', 'w') as f:
                json.dump(r, f, indent=2)
        else:
            valid = list(filter(lambda x: x >= 0, r))
            success = sum(valid)
            total = len(valid)

            rate1 = float(success) / len(r)
            rate2 = float(success) / total

            print("success rate (ALL) = {}, success rate (VALID) = {}, valid/invalid = {}/{}".format(rate1, rate2, total, len(r) - total))
