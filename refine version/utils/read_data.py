from itertools import count
from collections import namedtuple

KBIndex = namedtuple('KBIndex', ['ent_list', 'rel_list', 'rel_reverse_list', 'ent_id', 'rel_id', 'rel_reverse_id'])

def index_ent_rel(*filenames):
    ent_set = set()
    rel_set = set()
    rel_reverse = set()
    for filename in filenames:
        with open(filename) as f:
            for ln in f:
                s, r, t = ln.strip().split('\t')[:3]
                r_reverse = r + '_reverse'
                ent_set.add(s)
                ent_set.add(t)
                rel_set.add(r)
                rel_reverse.add(r_reverse)
    ent_list = sorted(list(ent_set))
    rel_list = sorted(list(rel_set))
    rel_reverse_list = sorted(list(rel_reverse))
 
    ent_id = dict(zip(ent_list, count()))
    rel_id = dict(zip(rel_list, count()))
    rel_size = len(rel_id)
    rel_reverse_id = dict(zip(rel_reverse_list, count(rel_size)))
    return KBIndex(ent_list, rel_list, rel_reverse_list, ent_id, rel_id, rel_reverse_id)


def graph_size(kb_index):
    return len(kb_index.ent_id), len(kb_index.rel_id)*2


def read_data(filename, kb_index):
    src = []
    rel = []
    dst = []
    with open(filename) as f:
        for ln in f:
            s, r, t = ln.strip().split('\t')
            src.append(kb_index.ent_id[s])
            rel.append(kb_index.rel_id[r])
            dst.append(kb_index.ent_id[t])
    return src, rel, dst

def read_data_triplets(filename, kb_index):
    triplets = []
    with open(filename) as f:
        for ln in f:
            s, r, t = ln.strip().split('\t')
            triplets.append([kb_index.ent_id[s], kb_index.rel_id[r], kb_index.ent_id[t]])
    return triplets

def read_reverse_data(filename, kb_index):
    src = []
    rel = []
    dst = []
    with open(filename) as f:
        for ln in f:
            s, r, t = ln.strip().split('\t')
            r_revsers = r + '_reverse'
            src.append(kb_index.ent_id[t])
            rel.append(kb_index.rel_id[r_revsers])
            dst.append(kb_index.ent_id[s])
    return src, rel, dst

def read_data_with_rel_reverse(filename, kb_index):
    src = []
    rel = []
    dst = []
    with open(filename) as f:
        for ln in f:
            s, r, t = ln.strip().split('\t')
            r_reverse = r + '_reverse'
            src.append(kb_index.ent_id[s])
            rel.append(kb_index.rel_id[r])
            dst.append(kb_index.ent_id[t])
            src.append(kb_index.ent_id[t])
            rel.append(kb_index.rel_reverse_id[r_reverse])
            dst.append(kb_index.ent_id[s])
    return src, rel, dst

def read_virtual_data(filename, kb_index):
    src = []
    rel = []
    dst = []
    with open(filename) as f:
        for ln in f:
            str = ln.strip().split('\t')
            s = kb_index.ent_id[str[0]]
            t = kb_index.ent_id[str[2]]
            r = kb_index.rel_id[str[1]]
            src.append(s)
            rel.append(r)
            dst.append(t)
    return src, rel, dst

def read_virtual_data_with_pre(filename, kb_index, lamda_dict):
    src = []
    rel = []
    dst = []
    presrc = []
    prerel = []
    predst = []
    lamda_id = []
    with open(filename) as f:
        for ln in f:
            str = ln.strip().split('\t')
            s = kb_index.ent_id[str[0]]
            t = kb_index.ent_id[str[2]]
            r = kb_index.rel_id[str[1]]
            src.append(s)
            rel.append(r)
            dst.append(t)

            lamda = str[3]
            lamda_id.append(lamda_dict[lamda])

            pres = kb_index.ent_id[str[4]]
            pret = kb_index.ent_id[str[6]]
            prer = kb_index.rel_id[str[5]]
            presrc.append(pres)
            prerel.append(prer)
            predst.append(pret)
    return src, rel, dst, presrc, prerel, predst, lamda_id

def read_virtual_data_dict(filename, kb_index):
    lamda_dict_head = {}
    lamda_dict_rel = {}
    lamda_dict_tail = {}
    lamda_dict = {}
    id = 0
    rule_ground_dict = {}
    with open(filename) as f:
        for ln in f:
            str = ln.strip().split('\t')
            s = kb_index.ent_id[str[0]]
            t = kb_index.ent_id[str[2]]
            r = kb_index.rel_id[str[1]]
            lamda = str[3]
            rule_s = kb_index.ent_id[str[4]]
            rule_r = kb_index.rel_id[str[5]]
            rule_t = kb_index.ent_id[str[6]]

            if lamda not in lamda_dict:
                lamda_dict[lamda] = id
                id = id + 1
            if lamda not in lamda_dict_head:
                lamda_dict_head[lamda] = list()
            if lamda not in lamda_dict_rel:
                lamda_dict_rel[lamda] = list()
            if lamda not in lamda_dict_tail:
                lamda_dict_tail[lamda] = list()
            lamda_dict_head[lamda].append(rule_s)
            lamda_dict_rel[lamda].append(rule_r)
            lamda_dict_tail[lamda].append(rule_t)
            rule_ground_dict[(s, r, t)] = ([rule_s], [rule_r], [rule_t], lamda)
    return lamda_dict_head, lamda_dict_rel, lamda_dict_tail, rule_ground_dict, lamda_dict
