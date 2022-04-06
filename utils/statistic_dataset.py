import os

def load_data(file_path):
    '''
        argument:
            file_path: ./data/FB15k-237

        return:
            entity2id, relation2id, train_triplets, valid_triplets, test_triplets
    '''
    with open(os.path.join(file_path, 'entity2id.txt')) as f:
        entity2id = dict()

        for line in f:
            entity, eid = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(file_path, 'relation2id.txt')) as f:
        relation2id = dict()

        for line in f:
            relation, rid = line.strip().split('\t')
            relation2id[relation] = int(rid)

    train_triplets = read_triplets(os.path.join(file_path, 'train.txt'), entity2id, relation2id)
    valid_triplets = read_triplets(os.path.join(file_path, 'valid.txt'), entity2id, relation2id)
    aux_triplets = read_triplets(os.path.join(file_path, '_aux.txt'), entity2id, relation2id)
    test_triplets = read_triplets(os.path.join(file_path, 'test.txt'), entity2id, relation2id)

    return entity2id, relation2id, train_triplets, valid_triplets, aux_triplets, test_triplets


def load_vn_data(file_path, triples_set, another_set):

    with open(os.path.join(file_path, 'entity2id.txt')) as f:
        entity2id = dict()

        for line in f:
            entity, eid = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(file_path, 'relation2id.txt')) as f:
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
    file_path_vn = os.path.join(file_path, 'vn.txt')
    with open(file_path_vn) as f:
        for line in f:
            grounding_list = line.strip().split('\t')
            grounding = eval(grounding_list[2])
            grounding_add = (entity2id[grounding[0]], relation2id[grounding[1]], entity2id[grounding[2]])
            if grounding_add not in triples_set:
                triples_set.add(grounding_add)
                cnt = cnt + 1
                if grounding_list[1] is "":
                    s1, r1, o1 = eval(grounding_list[0])
                    one_hop_pre.append((entity2id[s1], relation2id[r1], entity2id[o1]))
                    one_hop_ground.append(grounding_add)
                else:
                    s1, r1, o1 = eval(grounding_list[0])
                    s2, r2, o2 = eval(grounding_list[1])
                    two_hop_pre1.append((entity2id[s1], relation2id[r1], entity2id[o1]))
                    two_hop_pre2.append((entity2id[s2], relation2id[r2], entity2id[o2]))
                    two_hop_ground.append(grounding_add)

    triples_set.clear()
    file_path_sys = os.path.join(file_path, 'inner_rule.txt')
    with open(file_path_sys) as f:
        for line in f:
            list_line = eval(line)
            pre = list_line[0][0]
            ground = list_line[1][0]
            if pre[0] is 1:
                l, ps1, pr1, po1, ps2, pr2, po2, loc = ground
                ground1 = (entity2id[ps1], relation2id[pr1], entity2id[po1])
                ground2 = (entity2id[ps2], relation2id[pr2], entity2id[po2])
                g_list = [ground1, ground2]
                if g_list[loc - 1] not in triples_set:
                    cnt = cnt + 1
                    triples_set.add(g_list[loc - 1])
                    one_hop_ground.append(g_list[loc - 1])
            else:
                l, ps1, pr1, po1, ps2, pr2, po2, ps3, pr3, po3, loc = ground
                ground1 = (entity2id[ps1], relation2id[pr1], entity2id[po1])
                ground2 = (entity2id[ps2], relation2id[pr2], entity2id[po2])
                ground3 = (entity2id[ps3], relation2id[pr3], entity2id[po3])
                g_list = [ground1, ground2, ground3]
                if g_list[loc-1] not in triples_set:
                    cnt = cnt + 1
                    triples_set.add(g_list[loc - 1])
                    one_hop_ground.append(g_list[loc - 1])

    return one_hop_pre, one_hop_ground, two_hop_pre1, two_hop_pre2, two_hop_ground

def read_triplets(file_path, entity2id, relation2id):
    triplets = []

    with open(file_path) as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            triplets.append((entity2id[head], relation2id[relation], entity2id[tail]))

    return triplets

if __name__ == "__main__":
    sub_dataset = ["subject-5", "subject-10", "subject-15", "subject-20", "subject-25", "object-5", "object-10",
                   "object-15", "object-20", "object-25"]
    dataset = ["wn18", "wn18rr", "fb15k", "fb15k237", "yago37"]
    for dt in dataset:
        dir = os.path.join("../data", dt)
        for sdt in sub_dataset:
            target_dir = os.path.join(dir, sdt)
            entity2id, relation2id, train_triplets, valid_triplets, aux_triplets, test_triplets = load_data(target_dir)
            triples_set = set()
            triples_set.update(train_triplets)
            triples_set.update(aux_triplets)
            another_set = set(test_triplets)

            one_hop_pre, one_hop_ground, two_hop_pre1, two_hop_pre2, two_hop_ground = load_vn_data(target_dir,
                                                                                                   triples_set,
                                                                                                   another_set)
            seen_entities = set()
            unseen_entities = set()
            seen_neighbor = dict()
            dict_neighbor_observed = dict()
            dict_neighbor_before_vn = dict()
            dict_neighbor_after_vn = dict()
            seen_predict = dict()
            unseen_predict = dict()
            for s, r, t in train_triplets:
                seen_entities.add(s)
                seen_entities.add(t)
            for s, r, t in test_triplets:
                if s not in seen_entities:
                    unseen_entities.add(s)
                    if s not in unseen_predict:
                        unseen_predict[s] = 1
                    else:
                        unseen_predict[s] += 1
                else:
                    if s not in seen_predict:
                        seen_predict[s] = 1
                    else:
                        seen_predict[s] += 1

                if t not in seen_entities:
                    unseen_entities.add(t)
                    if t not in unseen_predict:
                        unseen_predict[t] = 1
                    else:
                        unseen_predict[t] += 1
                else:
                    if t not in seen_predict:
                        seen_predict[t] = 1
                    else:
                        seen_predict[t] += 1

            for s, r, t in aux_triplets:
                if s in unseen_entities:
                    if s not in dict_neighbor_before_vn:
                        dict_neighbor_before_vn[s] = 1
                    else:
                        dict_neighbor_before_vn[s] += 1
                    if s not in dict_neighbor_after_vn:
                        dict_neighbor_after_vn[s] = 1
                    else:
                        dict_neighbor_after_vn[s] += 1
                else:
                    if s not in dict_neighbor_observed:
                        if s in seen_predict:
                            dict_neighbor_observed[s] = 1
                    else:
                        if s in seen_predict:
                            dict_neighbor_observed[s] += 1
                if t in unseen_entities:
                    if t not in dict_neighbor_before_vn:
                        dict_neighbor_before_vn[t] = 1
                    else:
                        dict_neighbor_before_vn[t] += 1
                    if t not in dict_neighbor_after_vn:
                        dict_neighbor_after_vn[t] = 1
                    else:
                        dict_neighbor_after_vn[t] += 1
                else:
                    if t not in dict_neighbor_observed:
                        if t in seen_predict:
                            dict_neighbor_observed[t] = 1
                    else:
                        if t in seen_predict:
                            dict_neighbor_observed[t] += 1

            for s, r, t in one_hop_ground:
                if s in unseen_entities:
                    if s not in dict_neighbor_after_vn:
                        dict_neighbor_after_vn[s] = 1
                    else:
                        dict_neighbor_after_vn[s] += 1
                if t in unseen_entities:
                    if t not in dict_neighbor_after_vn:
                        dict_neighbor_after_vn[t] = 1
                    else:
                        dict_neighbor_after_vn[t] += 1

            for s, r, t in two_hop_ground:
                if s in unseen_entities:
                    if s not in dict_neighbor_after_vn:
                        dict_neighbor_after_vn[s] = 1
                    else:
                        dict_neighbor_after_vn[s] += 1
                if t in unseen_entities:
                    if t not in dict_neighbor_after_vn:
                        dict_neighbor_after_vn[t] = 1
                    else:
                        dict_neighbor_after_vn[t] += 1

            for s, r, t in train_triplets:
                if s in dict_neighbor_observed:
                    dict_neighbor_observed[s] += 1
                if t in dict_neighbor_observed:
                    dict_neighbor_observed[t] += 1

            before_sum = 0.0
            after_sum = 0.0
            before_ratio = 0.0
            after_ratio = 0.0
            observed_ratio = 0.0
            for key in dict_neighbor_before_vn.keys():
                before_sum = before_sum + dict_neighbor_before_vn[key]
                before_ratio = before_ratio + dict_neighbor_before_vn[key] / unseen_predict[key]
            for key in dict_neighbor_after_vn.keys():
                after_sum = after_sum + dict_neighbor_after_vn[key]
                after_ratio = after_ratio + dict_neighbor_after_vn[key] / unseen_predict[key]
            for key in dict_neighbor_observed.keys():
                if key in seen_predict:
                    observed_ratio = observed_ratio + dict_neighbor_observed[key] / seen_predict[key]
            '''
            print('num_train_triples: {}'.format(len(train_triplets)))
            print('num_aux_triples: {}'.format(len(aux_triplets)))
            print('num_unseen_entities: {}'.format(len(unseen_entities)))
            print('num_valid_triples: {}'.format(len(valid_triplets)))
            print('num_test_triples: {}'.format(len(test_triplets)))
            print('average neighbors before vn: {}'.format(before_sum/len(dict_neighbor_before_vn)))
            print('average neighbors after vn: {}'.format(after_sum/len(dict_neighbor_after_vn)))
            '''
            print('{0}-{1} {2} {3} {4} {5} {6} {7} {8} {9}'.format(dt, sdt, len(train_triplets), len(aux_triplets), len(unseen_entities),
                                                                                len(valid_triplets), len(test_triplets),
                                                                                before_ratio/len(dict_neighbor_before_vn), after_ratio/len(dict_neighbor_after_vn),
                                                                   observed_ratio/len(dict_neighbor_observed)))