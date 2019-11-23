import json
from nltk.stem import WordNetLemmatizer
from multiprocessing import Pool
import multiprocessing
import argparse
import Constants
import random


def eq(graph_entity, query, value=False):
    if query is None:
        return True
    else:
        if graph_entity == query:
            return True
        elif graph_entity in Constants.hypernym.get(query, [query]):
            return True
        elif ' ' in graph_entity and ' ' in query:
            return graph_entity.split(' ')[-1] == query.split(' ')[-1]
        elif ' ' in graph_entity:
            return query in graph_entity.split(' ')
        elif ' ' in query:
            return graph_entity in query.split(' ')
        else:
            return False


APIS = {}
APIS['relate'] = lambda graph, candidates, relation: relate(graph, candidates, relation, None, False)
APIS['relate_inv'] = lambda graph, candidates, relation: relate(graph, candidates, relation, None, True)
APIS['relate_name'] = lambda graph, candidates, relation, name: relate(graph, candidates, relation, name, False)
APIS['relate_inv_name'] = lambda graph, candidates, relation, name: relate(graph, candidates, relation, name, True)


def relate(graph, candidates, relation, name, reverse):
    targets = []
    if reverse:
        for cand in candidates:
            for thing in graph:
                if eq(graph[thing]['name'], name):
                    for elem in graph[thing]['relations']:
                        if elem['name'] == relation and elem['object'] == cand:
                            targets.append(thing)
                            break
    else:
        for cand in candidates:
            for elem in graph[cand]['relations']:
                if eq(graph[elem['object']]['name'], name) and elem['name'] == relation:
                    targets.append(elem['object'])

    return targets


APIS['select'] = lambda graph, name: select(graph, name)


def select(graph, name):
    targets = []
    for k, v in graph.items():
        if eq(v['name'], name) or name == 'this':
            targets.append(k)
    return targets


APIS['relate_attr'] = lambda graph, candidates, relate, name: relate_attr(graph, candidates, name)


def relate_attr(graph, candidates, name):
    targets = []
    for cand in candidates:
        own = set(graph[cand]['attributes'])
        for thing in graph:
            left = own & set(graph[thing]['attributes'])
            if left:
                targets.append(thing)
    return targets


APIS['filter'] = lambda graph, candidates, name: filter_attr(graph, candidates, name, None, False)
APIS['filter_not'] = lambda graph, candidates, name: filter_attr(graph, candidates, name, None, True)
APIS['filter_h'] = lambda graph, candidates, name: filter_attr(graph, candidates, name, 'h', False)
#APIS['filter_v'] = lambda graph, candidates, name: filter_attr(graph, candidates, name, 'v', False)


def filter_attr(graph, candidates, name, position, neg):
    targets = []
    if position:
        max_x, max_y = graph['0']['w'], graph['0']['h']
        if position == 'h':
            for cand in candidates:
                if name == 'middle':
                    targets.append(cand)
                else:
                    if graph[cand]['x'] < max_x // 2:
                        true_pos = 'left'
                    else:
                        true_pos = 'right'
                    if name == true_pos:
                        targets.append(cand)

            for cand in candidates:
                if name == 'middle':
                    targets.append(cand)
                else:
                    if graph[cand]['y'] < max_y // 2:
                        true_pos = 'top'
                    else:
                        true_pos = 'bottom'
                    if name == true_pos:
                        targets.append(cand)
    else:
        for cand in candidates:
            if neg:
                if name not in graph[cand]['attributes']:
                    targets.append(cand)
            else:
                if name in graph[cand]['attributes']:
                    targets.append(cand)
    return targets


APIS['query_n'] = lambda graph, candidates: query(graph, candidates, 'n', None)
APIS['query_h'] = lambda graph, candidates: query(graph, candidates, 'h', None)
#APIS['query_v'] = lambda graph, candidates: query(graph, candidates, 'v', None)
APIS['query'] = lambda graph, candidates, attr: query(graph, candidates, None, attr)
APIS['query_f'] = lambda graph, attr: query(graph, None, 'f', attr)


def query(graph, candidates, category, attr):
    if candidates is not None and len(candidates) == 0:
        return None

    if category == 'n':
        return graph[candidates[0]]['name']
    elif category == 'h':
        max_x, max_y = graph['0']['w'], graph['0']['h']
        if graph[candidates[0]]['x'] > max_x // 2:
            return 'right'
        else:
            return 'left'
    elif category == 'v':
        max_x, max_y = graph['0']['w'], graph['0']['h']
        if graph[candidates[0]]['y'] > max_y // 2:
            return 'bottom'
        else:
            return 'top'
    elif attr in Constants.ONTOLOGY:
        potential_attr = set(Constants.ONTOLOGY[attr])
        if category == 'f':
            potential = set(graph['0']['attributes']) & set(potential_attr)
            if potential:
                return list(potential)[0]
        else:
            for cand in candidates:
                potential = set(graph[cand]['attributes']) & potential_attr
                if potential:
                    return list(potential)[0]
        return None
    else:
        for cand in candidates:
            if graph[cand]['attributes']:
                return graph[cand]['attributes'][0]
        return None


APIS['verify'] = lambda graph, candidates, val: verify(graph, candidates, val, None)
APIS['verify_f'] = lambda graph, val: verify(graph, None, val, 'f')
APIS['verify_h'] = lambda graph, candidates, val: verify(graph, candidates, val, 'h')
APIS['verify_v'] = lambda graph, candidates, val: verify(graph, candidates, val, 'v')


def verify(graph, candidates, val, category):
    if candidates is not None and len(candidates) == 0:
        return False, None

    if category is None:
        for cand in candidates:
            if val in graph[cand]['attributes']:
                return True, cand
        return False, None
    elif category == 'f':
        if val in graph['0']['attributes']:
            return True, '0'
        else:
            return False, None
    elif category == 'h':
        max_x, max_y = graph['0']['w'], graph['0']['h']
        if graph[candidates[0]]['x'] > max_x // 2:
            pos = 'right'
        else:
            pos = 'left'
        if pos == val:
            return True, candidates[0]
        else:
            return False, None
    elif category == 'v':
        max_x, max_y = graph['0']['w'], graph['0']['h']
        if graph[candidates[0]]['y'] > max_y // 2:
            pos = 'bottom'
        else:
            pos = 'top'
        if pos == val:
            return True, candidates[0]
        else:
            return False, None


APIS['verify_rel'] = lambda graph, candidates, relation, name: verify_relation(graph, candidates, relation, name, False)
APIS['verify_rel_inv'] = lambda graph, candidates, relation, name: verify_relation(
    graph, candidates, relation, name, True)


def verify_relation(graph, candidates, relation, name, reverse):
    if len(candidates) == 0:
        return False, None

    if reverse:
        for cand in candidates:
            for thing in graph:
                if eq(graph[thing]['name'], name):
                    for elem in graph[thing]['relations']:
                        if elem['name'] == relation and elem['object'] == cand:
                            return True, thing
                        # elif elem['name'] != relation and elem['object'] == cand:
                        #    return False, None
    else:
        for cand in candidates:
            for elem in graph[cand]['relations']:
                if eq(graph[elem['object']]['name'], name) and elem['name'] == relation:
                    return True, cand
                # elif eq(graph[elem['object']]['name'], name) and elem['name'] != relation:
                #    return False, None

    return False, None


APIS['choose_n'] = lambda graph, candidates, name1, name2: choose(graph, candidates, name1, name2, 'n')
APIS['choose_f'] = lambda graph, attr1, attr2: choose(graph, None, attr1, attr2, 'f')
APIS['choose'] = lambda graph, candidates, attr1, attr2: choose(graph, candidates, attr1, attr2, 'a')
APIS['choose_attr'] = lambda graph, candidates, attr, attr1, attr2: choose(graph, candidates, attr1, attr2, 'a')


def choose(graph, candidates, val1, val2, category):
    if candidates is not None and len(candidates) == 0:
        return None

    if category == 'n':
        for cand in candidates:
            if eq(graph[cand]['name'], val1):
                return val1
            elif eq(graph[cand]['name'], val2):
                return val2

        return None
    elif category == 'f':
        if val1 in graph['0']['attributes']:
            return val1
        elif val2 in graph['0']['attributes']:
            return val2

        return None
    else:
        for cand in candidates:
            if val1 in graph[cand]['attributes']:
                return val1
            elif val1 in graph[cand]['attributes']:
                return val2

        return None


APIS['choose_h'] = lambda graph, candidates, position1, position2: choose_pos(graph, candidates, 'h')
APIS['choose_v'] = lambda graph, candidates, position1, position2: choose_pos(graph, candidates, 'v')


def choose_pos(graph, candidates, category):
    if len(candidates) == 0:
        return None

    if category == 'h':
        max_x, max_y = graph['0']['w'], graph['0']['h']
        if graph[candidates[0]]['x'] > max_x // 2:
            return 'right'
        else:
            return 'left'
    elif category == 'v':
        max_x, max_y = graph['0']['w'], graph['0']['h']
        if graph[candidates[0]]['y'] > max_y // 2:
            return 'bottom'
        else:
            return 'top'

    return None


APIS['choose_rel_inv'] = lambda graph, candidates, name, relation1, relation2: choose_rel(
    graph, candidates, name, relation1, relation2, True)
#APIS['choose_rel_inv'] = lambda graph, candidates, name, relation1, relation2: choose_rel(graph, candidates, name, relation1, relation2, False)


def choose_rel(graph, candidates, name, relation1, relation2, reverse):
    if len(candidates) == 0:
        return None

    for cand in candidates:
        for thing in graph:
            if eq(graph[thing]['name'], name):
                for elem in graph[thing]['relations']:
                    if elem['name'] == relation1 and elem['object'] == cand:
                        return relation1
                    elif elem['name'] == relation2 and elem['object'] == cand:
                        return relation2
    return None


APIS['choose_subj'] = lambda graph, candidate1, candidate2, attr: choose_attr(graph, candidate1, candidate2, attr)


def choose_attr(graph, candidate1, candidate2, attr):
    return None


APIS['exist'] = lambda graph, candidates: exist(candidates)


def exist(candidates):
    if len(candidates) > 0:
        return True, candidates[0]
    else:
        return False, None


APIS['or'] = lambda graph, candidate1, candidate2: binary(candidate1, candidate2, 'or')
APIS['and'] = lambda graph, candidate1, candidate2: binary(candidate1, candidate2, 'and')


def binary(candidate1, candidate2, logic):
    if logic == 'or':
        return candidate1 or candidate2, None
    elif logic == 'and':
        return candidate1 and candidate2, None


APIS['different'] = lambda graph, candidates: compare(graph, candidates, 'different')
APIS['same'] = lambda graph, candidates: compare(graph, candidates, 'same')


def compare(graph, candidates, category):
    if len(candidates) < 2:
        return None

    names = []
    for cand in candidates:
        names.append(graph[cand]['name'])
    uniq_names = set(names)

    if category == 'different':
        return len(uniq_names) > 1, None
    elif category == 'same':
        return len(uniq_names) == 1, None


"""
APIS['same_v2'] = lambda graph, candidate1, candidate2: compare_v2(graph, candidate1, candidate2, 'same')
APIS['different_v2'] = lambda graph, candidate1, candidate2: compare_v2(graph, candidate1, candidate2, 'different')
def compare_v2(graph, candidate1, candidate2, category):
	if len(candidate1) == 0 or len(candidate2) == 0:
		return None

	name1 = graph[candidate1[0]]['name']
	name2 = graph[candidate2[0]]['name']
	if category == 'different':
		return name1 != name2
	elif category == 'same':
		return name1 == name2
"""
APIS['common'] = lambda graph, candidate1, candidate2: common(graph, candidate1, candidate2)


def common(graph, candidate1, candidate2):
    if len(candidate1) == 0 or len(candidate2) == 0:
        return None

    attr1 = []
    for cand in candidate1:
        attr1.extend(graph[candidate1[0]]['attributes'])
    attr2 = []
    for cand in candidate2:
        attr2.extend(graph[candidate2[0]]['attributes'])
    attr = list(set(attr1) & set(attr2))
    for a in attr:
        if a in Constants.ONTOLOGY['color']:
            return "color"
        else:
            return "material"
    return None


APIS['same_attr'] = lambda graph, candidate1, candidate2, attr: compare_color(
    graph, attr, candidate1, candidate2, 'same')
APIS['different_attr'] = lambda graph, candidate1, candidate2, attr: compare_color(
    graph, attr, candidate1, candidate2, 'different')


def compare_color(graph, attr, candidate1, candidate2, category):
    if len(candidate1) == 0 or len(candidate2) == 0:
        return None

    for cand in candidate1:
        for color in Constants.ONTOLOGY[attr]:
            if color in graph[cand]['attributes']:
                color1 = color
    for cand in candidate2:
        for color in Constants.ONTOLOGY[attr]:
            if color in graph[cand]['attributes']:
                color2 = color

    if category == "same":
        return color1 == color2, None
    else:
        return color1 != color2, None
