import numpy as np
import os, sys

sys.path.append(os.getcwd())
from utils.constants import *
from utils.helpers import *


def fact_format(fact):
    """
        Formatting the input fact according to rules
    """
    new_fact = {}
    if 'question-category' in fact:
        new_fact['question-category'] = fact['question-category']
    else:
        new_fact['question-category'] = 'How'
    if 'question-type' in fact:
        new_fact['question-type'] = fact['question-type']
    else:
        new_fact['question-type'] = 'summary'
    # time
    if 'time' in fact and fact['time']:
        new_fact['time'] = {}
        if len(fact['time']) == 1:
            new_fact['time']['len'] = "1"
        else:
            new_fact['time']['len'] = '>1'
        new_fact['time']['type'] = fact['time'][0]['type']
        if 'role' in fact['time'][0].keys():
            temp_role = fact['time'][0]['role']
            if temp_role == "none" or temp_role == "":
                new_fact['time']['role'] = None
            else:
                new_fact['time']['role'] = temp_role
        else:
            new_fact['time']['role'] = None
    else:
        new_fact['time'] = None
    # measure
    if 'measure' in fact and fact['measure']:
        new_fact['measure'] = {}
        if len(fact['measure']) == 1:
            new_fact['measure']['len'] = "1"
            new_fact['measure']['aggregation'] = fact['measure'][0]['aggregation']
            new_fact['measure']['type'] = fact['measure'][0]['type']
        else:
            new_fact['measure']['len'] = '>1'
            new_fact['measure']['aggregation'] = 'multiple'
            new_fact['measure']['type'] = 'multiple'
    else:
        new_fact['measure'] = None
    # subspace
    if 'subspace' in fact and fact['subspace']:
        new_fact['subspace'] = {}
        if len(fact['subspace']) == 1:
            new_fact['subspace']['len'] = "1"
            new_fact['subspace']['type'] = fact['subspace'][0]['type']
        else:
            new_fact['subspace']['len'] = '>1'
            new_fact['subspace']['type'] = 'multiple'
    else:
        new_fact['subspace'] = None
    # focus
    if 'focus' in fact and fact['focus']:
        new_fact['focus'] = {}
        if len(fact['focus']) == 1:
            new_fact['focus']['len'] = "1"
            new_fact['focus']['value'] = fact['focus'][0]['value']
        else:
            new_fact['focus']['len'] = '>1'
            new_fact['focus']['value'] = 'multiple'
    else:
        new_fact['focus'] = None
    return new_fact


def get_rules(node, parentkey, rules):
    current_rule = parentkey + ' -> ' + ' "+" '.join(sorted(node.keys()))
    rules.append(current_rule)
    for k in sorted(node.keys()):
        v = node[k]
        if type(v) is dict:
            get_rules(v, k, rules)
        else:
            temp_rule = k + ' -> ' + '"' + str(v) + '"'
            if temp_rule not in rules:
                rules.append(temp_rule)


@memoize
def get_total_rules():
    """
        Get all the rules.
        Return:
            `rules`: list of all rules.
            `rule2index`: the index corresponding to each rule.
    """
    rule_path = os.path.join(os.path.dirname(__file__), 'structural_rules.txt')
    rules = []
    with open(rule_path, 'r') as inputs:
        for line in inputs:
            line = line.strip()
            rules.append(line)
    rule2index = {}
    for i, r in enumerate(rules):
        rule2index[r] = i
    return rules, rule2index


def extract_structual_info(fact):
    """
        Extract the structure information in item and represent the rules in the structure as one-hot vectors.
    """
    total_rules, rule2index = get_total_rules()
    fact_rules = []
    fact = fact_format(fact)
    get_rules(fact, 'root', fact_rules)

    one_hot = np.zeros((MAX_STRUCT_FEATURE_LEN, len(total_rules)), dtype=np.float32)
    for r in fact_rules:
        if r not in rule2index:
            print("Missing:", r)
    indices = [rule2index[r] for r in fact_rules]
    one_hot[np.arange(len(indices)), indices] = 1
    one_hot[np.arange(len(indices), MAX_STRUCT_FEATURE_LEN), -1] = 1

    return one_hot