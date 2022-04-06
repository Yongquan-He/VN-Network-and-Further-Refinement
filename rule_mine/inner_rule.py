import csv
import os

def get_ground(current_rule, original_triples_path, exist_triples, ground_set, flag):
    old_rule = current_rule
    if len(old_rule) == 7:
        if flag is 1:
            current_rule = [old_rule[4], old_rule[5], old_rule[6], old_rule[3], old_rule[0], old_rule[1], old_rule[2]]
        else:
            current_rule = old_rule
    else:
        if flag is 2:
            current_rule = [old_rule[0], old_rule[1], old_rule[2], old_rule[7], old_rule[8], old_rule[9], old_rule[6], old_rule[3], old_rule[4], old_rule[5]]
        elif flag is 1:
            current_rule = [old_rule[7], old_rule[8], old_rule[9], old_rule[3], old_rule[4], old_rule[5], old_rule[6], old_rule[0], old_rule[1], old_rule[2]]
        else:
            current_rule = old_rule
    rule_var = []  # variables in current rule
    match_var = [2] * 6  # if match_var[i] == match_var[j] then rule_var[i] == rule_var[j]

    # initialize var0 and var1
    match_var[0] = 0
    match_var[1] = 1

    # extract variables and relations from rule

    if len(current_rule) == 7:  # body size == 1
        for i in (0, 2, 4, 6):
            rule_var.append(current_rule[i])

        body_relation1 = current_rule[1]
        body_relation2 = None
        head_relation = current_rule[5]

        # variables matching
        for i in range(2, 4):
            if rule_var[i] == rule_var[0]:
                match_var[i] = 0
        for i in range(2, 4):
            if rule_var[i] == rule_var[1]:
                match_var[i] = 1

    else:  # body size == 2
        for i in (0, 2, 3, 5, 7, 9):
            rule_var.append(current_rule[i])

        body_relation1 = current_rule[1]
        body_relation2 = current_rule[4]
        head_relation = current_rule[8]

        # variables matching
        for i in range(2, 6):
            if rule_var[i] == rule_var[0]:
                match_var[i] = 0
            if rule_var[i] == rule_var[1]:
                match_var[i] = 1

    # search for facts with body relations in original facts
    # also search for facts with head relations in original facts, to check duplicate
    original_facts = open(original_triples_path, 'r', encoding='utf-8')
    facts_reader = csv.reader(original_facts, delimiter='\t')
    br1 = []
    br2 = []
    hr = []

    for row_fact in facts_reader:
        if row_fact[1] == body_relation1:
            br1.append(row_fact)
        if row_fact[1] == head_relation:
            hr.append(row_fact)
        if row_fact[1] == body_relation2 and body_relation2:  # body size == 2
            br2.append(row_fact)

    # check and store potential new facts
    candidate_new_facts = []
    candidate_new_body1 = []
    if not body_relation2:  # body size == 1
        for facts in br1:
            if match_var[0] == match_var[2]:
                current_new_fact = [facts[0], head_relation, facts[2]]
            else:
                current_new_fact = [facts[2], head_relation, facts[0]]

            candidate_new_body1.append(facts)
            candidate_new_facts.append(current_new_fact)
            if (current_new_fact[0], current_new_fact[1], current_new_fact[2]) not in exist_triples:
                if flag is 1:
                    ground_set.add((1, current_new_fact[0], current_new_fact[1], current_new_fact[2], facts[0], facts[1], facts[2], flag))
                else:
                    ground_set.add((1, facts[0], facts[1], facts[2], current_new_fact[0], current_new_fact[1], current_new_fact[2], flag))

    else:  # body size == 2
        for facts1 in br1:
            for facts2 in br2:
                current_match = [0, 1, 2, 2]
                # extract the current pattern
                current_var = [facts1[0], facts1[2], facts2[0], facts2[2]]
                for i in range(2, 4):
                    if current_var[i] == current_var[0]:
                        current_match[i] = 0
                    if current_var[i] == current_var[1]:
                        current_match[i] = 1

                # if the pattern matched
                if current_match == match_var[0:4]:
                    head_var1_index = match_var.index(match_var[4])
                    head_var2_index = match_var.index(match_var[5])

                    current_new_fact = [current_var[head_var1_index], head_relation, current_var[head_var2_index]]

                    current_rule_instance = []
                    current_rule_instance.extend(facts1)
                    current_rule_instance.extend(facts2)
                    current_rule_instance.extend(current_new_fact)
                    if (current_new_fact[0], current_new_fact[1], current_new_fact[2]) not in exist_triples:
                        if flag is 1:
                            ground_set.add((2, current_new_fact[0], current_new_fact[1], current_new_fact[2], facts2[0], facts2[1], facts2[2],
                                            facts1[0], facts1[1], facts1[2], flag))

                        elif flag is 2:
                            ground_set.add((2, facts1[0], facts1[1], facts1[2], current_new_fact[0], current_new_fact[1], current_new_fact[2], facts2[0],
                                            facts2[1], facts2[2], flag))
                        else:
                            ground_set.add((
                                           2, facts1[0], facts1[1], facts1[2], facts2[0],
                                           facts2[1], facts2[2], current_new_fact[0], current_new_fact[1],
                                           current_new_fact[2], flag))
    return ground_set

def get_sys(mined_rules_path, original_triples_path, exist_triples):
    sys_pre_list = []
    sys_ground_list = []

    with open(mined_rules_path, 'r',  encoding='utf-8') as rules:
        rule_reader = csv.reader(rules, delimiter='\t')
        # skip the header
        next(rule_reader, None)
        for row_rule in rule_reader:
            set_pre = set()
            set_ground = set()
            current_rule = row_rule[1].split()
            if len(current_rule) is 7:
                set_pre = get_ground(current_rule, original_triples_path, exist_triples, set_pre, 0)
                set_ground = get_ground(current_rule, original_triples_path, exist_triples, set_ground, 1)
            else:
                set_pre = get_ground(current_rule, original_triples_path, exist_triples, set_pre, 0)
                set_ground = get_ground(current_rule, original_triples_path, exist_triples, set_ground, 1)
                set_ground = get_ground(current_rule, original_triples_path, exist_triples, set_ground, 2)

            sys_pre_list.append(set_pre)
            sys_ground_list.append(set_ground)
    return sys_pre_list, sys_ground_list

def load_data(file_path):
    relation_set = set()
    '''
        argument:
            file_path: ./data/FB15k-237

        return:
            entity2id, relation2id, train_triplets, valid_triplets, test_triplets
    '''
    with open(os.path.join(file_path, 'entity2id.txt'), encoding='utf-8') as f:
        entity2id = dict()

        for line in f:
            entity, eid = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(file_path, 'relation2id.txt'), encoding='utf-8') as f:
        relation2id = dict()

        for line in f:
            relation, rid = line.strip().split('\t')
            relation2id[relation] = int(rid)
            relation_set.add(relation)

    train_triplets = read_triplets(os.path.join(file_path, 'train.txt'), entity2id, relation2id)
    valid_triplets = read_triplets(os.path.join(file_path, 'valid.txt'), entity2id, relation2id)
    aux_triplets = read_triplets(os.path.join(file_path, '_aux.txt'), entity2id, relation2id)
    test_triplets = read_triplets(os.path.join(file_path, 'test.txt'), entity2id, relation2id)

    return entity2id, relation_set, train_triplets, valid_triplets, aux_triplets, test_triplets

def read_triplets(file_path, entity2id, relation2id):
    triplets = []

    with open(file_path, encoding='utf-8') as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            triplets.append((head, relation, tail))

    return triplets

def get_path(sr2o, or2s, sys_set, sys, relation_set, path_confidence):
    if sys[0] is 1:
        l, s1, r1, o1, s2, r2, o2, loc = sys
        list2 = [s1, r1, o1, s2, r2, o2]
        for sys_p in sys_set:
            has_done = set()
            l, ps1, pr1, po1, ps2, pr2, po2, loc = sys_p
            list_p = [ps1, pr1, po1, ps2, pr2, po2]
            for ps, s in zip(list_p, list2):
                if s not in has_done:
                    p = get_path_and_confidence(sr2o, or2s, ps, s, relation_set, path_confidence)
                    if len(p) > 0:
                        return [[sys_p], [sys], p]
                has_done.add(s)
    else:
        l, s1, r1, o1, s2, r2, o2, s3, r3, o3, loc = sys
        list2 = [s1, r1, o1, s2, r2, o2, s3, r3, o3]
        for sys_p in sys_set:
            has_done = set()
            l, ps1, pr1, po1, ps2, pr2, po2, ps3, pr3, po3, loc = sys_p
            list_p = [ps1, pr1, po1, ps2, pr2, po2, ps3, pr3, po3]
            for ps, s in zip(list_p, list2):
                if s not in has_done:
                    p = get_path_and_confidence(sr2o, or2s, ps, s, relation_set, path_confidence)
                    if len(p) > 0:
                        return [[sys_p], [sys], p]
                has_done.add(s)
    return []

def get_path_and_confidence(sr2o, or2s, ps, s, relation_set, path_confidence):
    pc = path_confidence
    for rel in relation_set:
        p = 1.0
        if (ps, rel) in sr2o and (s, rel) in sr2o:
            mid_set1 = (sr2o[(ps, rel)] & sr2o[(s, rel)])
            for mid in mid_set1:
                c = p / len(sr2o[(s, rel)]) / len(or2s[(mid, rel)])
                if c >= pc:
                    return [ps, rel, mid, rel + "_inverse", s, c]
        if (ps, rel) in or2s and (s, rel) in or2s:
            mid_set2 = (or2s[(ps, rel)] & or2s[(s, rel)])
            for mid in mid_set2:
                c = p / len(or2s[(s, rel)]) / len(sr2o[(mid, rel)])
                if c >= pc:
                    return [ps, rel + "_inverse", mid, rel, s, c]
        if (ps, rel) in sr2o:
            if s in sr2o[(ps, rel)]:
                c = p / len(sr2o[(ps, rel)])
                if c >= pc:
                    return [ps, rel, s, c]
        if (ps, rel) in or2s:
            if s in or2s[(ps, rel)]:
                c = p / len(or2s[(ps, rel)])
                if c >= pc:
                    return [ps, rel + "_inverse", s, c]
    return []

def load_vn_data(file_path, triples_set, another_set):
    with open(os.path.join(file_path, 'entity2id.txt'), encoding='utf-8') as f:
        entity2id = dict()

        for line in f:
            entity, eid = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(file_path, 'relation2id.txt'), encoding='utf-8') as f:
        relation2id = dict()

        for line in f:
            relation, rid = line.strip().split('\t')
            relation2id[relation] = int(rid)

    one_hop_pre = []
    one_hop_ground = []

    two_hop_pre1 = []
    two_hop_pre2 = []
    two_hop_ground = []
    cnt = 0
    n_cnt = 0
    file_path_vn = os.path.join(file_path, 'vn.txt')
    with open(file_path_vn, encoding='utf-8') as f:
        for line in f:
            grounding_list = line.strip().split('\t')
            grounding = eval(grounding_list[2])
            grounding_add = (grounding[0], grounding[1], grounding[2])
            if grounding_add in another_set:
                n_cnt = n_cnt + 1
            if grounding_add not in triples_set:
                cnt = cnt + 1
                if grounding_list[1] is "":
                    if len(eval(grounding_list[0])) is 3:
                        s1, r1, o1 = eval(grounding_list[0])
                        one_hop_pre.append((entity2id[s1], relation2id[r1], entity2id[o1]))
                        one_hop_ground.append(grounding_add)
                else:
                    if len(eval(grounding_list[0])) is 3 and len(eval(grounding_list[1])) is 3:
                        s1, r1, o1 = eval(grounding_list[0])
                        s2, r2, o2 = eval(grounding_list[1])
                        two_hop_pre1.append((entity2id[s1], relation2id[r1], entity2id[o1]))
                        two_hop_pre2.append((entity2id[s2], relation2id[r2], entity2id[o2]))
                        two_hop_ground.append(grounding_add)

    return one_hop_pre, one_hop_ground, two_hop_pre1, two_hop_pre2, two_hop_ground

def count(file_path, another_set):

    with open(os.path.join(file_path, 'entity2id.txt'), encoding='utf-8') as f:
        entity2id = dict()

        for line in f:
            entity, eid = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(file_path, 'relation2id.txt'), encoding='utf-8') as f:
        relation2id = dict()

        for line in f:
            relation, rid = line.strip().split('\t')
            relation2id[relation] = int(rid)

    one_hop_pre = []
    one_hop_ground = []

    two_hop_pre1 = []
    two_hop_pre2 = []
    two_hop_ground = []
    n_cnt = 0
    file_path_vn = os.path.join(file_path, 'vn.txt')
    with open(file_path_vn, encoding='utf-8') as f:
        for line in f:
            grounding_list = line.strip().split('\t')
            grounding = eval(grounding_list[2])
            grounding_add = (grounding[0], grounding[1], grounding[2])
            if grounding_add in another_set:
                n_cnt = n_cnt + 1
                triples_set.add(grounding_add)
                another_set.add(grounding_add)
    print("Triples in test dataset: " + str(n_cnt))

    sys_one_pre1 = []
    sys_one_pre2 = []
    sys_one_ground1 = []
    sys_one_ground2 = []
    sys_one_confidence = []
    sys_one_loc = []

    sys_two_pre1 = []
    sys_two_pre2 = []
    sys_two_pre3 = []
    sys_two_ground1 = []
    sys_two_ground2 = []
    sys_two_ground3 = []
    sys_two_confidence = []
    sys_two_loc = []

    file_path_sys = os.path.join(file_path, 'inner_rule.txt')
    with open(file_path_sys, encoding='utf-8') as f:
        for line in f:
            list_line = eval(line)
            pre = list_line[0][0]
            ground = list_line[1][0]
            path = list_line[2][0]
            if pre[0] is 1:
                l, s1, r1, o1, s2, r2, o2, loc = pre
                pre1 = (s1, r1, o1)
                pre2 = (s2, r2, o2)
                l, ps1, pr1, po1, ps2, pr2, po2, loc = ground
                ground1 = (ps1, pr1, po1)
                ground2 = (ps2, pr2, po2)
                g_list = [ground1, ground2]
                if g_list[loc - 1] in another_set and g_list[loc - 1] not in triples_set:
                    n_cnt = n_cnt + 1

                sys_one_pre1.append(pre1)
                sys_one_pre2.append(pre2)
                sys_one_ground1.append(ground1)
                sys_one_ground2.append(ground2)
                sys_one_confidence.append(path[len(path) - 1])
                sys_one_loc.append(loc)
            else:
                l, s1, r1, o1, s2, r2, o2, s3, r3, o3, loc = pre
                pre1 = (s1, r1, o1)
                pre2 = (s2, r2, o2)
                pre3 = (s3, r3, o3)
                l, ps1, pr1, po1, ps2, pr2, po2, ps3, pr3, po3, loc = ground
                ground1 = (ps1, pr1, po1)
                ground2 = (ps2, pr2, po2)
                ground3 = (ps3, pr3, po3)
                g_list = [ground1, ground2, ground3]
                if g_list[loc-1] in another_set:
                    n_cnt = n_cnt + 1

                    sys_two_pre1.append(pre1)
                    sys_two_pre2.append(pre2)
                    sys_two_pre3.append(pre3)
                    sys_two_ground1.append(ground1)
                    sys_two_ground2.append(ground2)
                    sys_two_ground3.append(ground3)
                    sys_two_confidence.append(path[len(path)-1])
                    sys_two_loc.append(loc)
    print("Triples in test dataset: " + str(n_cnt))
    return one_hop_pre, one_hop_ground, two_hop_pre1, two_hop_pre2, two_hop_ground

if __name__ == "__main__":
    sub_dataset = ["subject-5", "subject-10", "subject-15", "subject-20", "subject-25", "object-5", "object-10", "object-15", "object-20", "object-25"]
    dataset = ["wn18", "wn18rr", "fb15k", "fb15k237", "yago37"]
    path_confidence = 0.0
    for dt in dataset:
        dir = os.path.join("../data", dt)
        for sdt in sub_dataset:
            target_dir = os.path.join(dir, sdt)
            with open(os.path.join(target_dir, "inner_rule.txt"), 'w', newline='', encoding='utf-8') as target:
                entity2id, relation_set, train_triplets, valid_triplets, aux_triplets, test_triplets = load_data(
                    target_dir)
                triples_set = set()
                triples_set.update(train_triplets)
                triples_set.update(valid_triplets)
                triples_set.update(aux_triplets)
                another_set = set(test_triplets)
                one_hop_pre, one_hop_ground, two_hop_pre1, two_hop_pre2, two_hop_ground = load_vn_data(target_dir,
                                                                                                       triples_set,
                                                                                                       another_set)
                all_graph = list()
                all_graph.extend(train_triplets)
                all_graph.extend(valid_triplets)
                all_graph.extend(aux_triplets)
                sr2o = {}
                or2s = {}
                for triple in all_graph:
                    s, r, o = triple
                    if (s, r) not in sr2o:
                        sr2o[(s, r)] = set()
                    if (o, r) not in or2s:
                        or2s[(o, r)] = set()
                    sr2o[(s, r)].add(o)
                    or2s[(o, r)].add(s)
                if dt is "yago37":
                    sys_pre_list, sys_ground_list = get_sys(os.path.join(target_dir, "pca_sorted_rule.txt"),
                                                            os.path.join(target_dir, "_aux.txt"),
                                                            triples_set)
                else:
                    sys_pre_list, sys_ground_list = get_sys(os.path.join(target_dir, "pca_sorted_rule_pool.txt"),
                                                            os.path.join(target_dir, "_aux.txt"),
                                                            triples_set)
                all_inner = 0
                symmetric_inner = 0
                for sys_pre_set, sys_ground_set in zip(sys_pre_list, sys_ground_list):
                    for sys_ground in sys_ground_set:
                        ret = get_path(sr2o, or2s, sys_pre_set, sys_ground, relation_set, path_confidence)
                        if len(ret) > 0:
                            target.write(str(ret)+"\n")
                            if len(ret[2]) > 4:
                                symmetric_inner += 1
                            all_inner += 1
                print("all inner: " + str(all_inner))
                print("symmetric inner: " + str(symmetric_inner))
                count(target_dir, another_set)
