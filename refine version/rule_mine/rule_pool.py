import argparse
import os
import csv
import pandas

def derive_new_triples(mined_rules_path, original_triples_path, new_triples_path):
    with open(new_triples_path, 'w', newline='') as target:
        new_writer = csv.writer(target, delimiter='\t')
        with open(mined_rules_path, 'r') as rules:
            rule_reader = csv.reader(rules, delimiter='\t')
            # skip the header
            next(rule_reader, None)
            for row_rule in rule_reader:
                current_rule = row_rule[1].split()

                # extract current rule's relation and variable pattern
                # if body size == 1: e.g. ?a Conjunction ?b => ?a Concession ?b
                # len == 7, body relation1 index == 1, head relation index = 5
                # if body size == 2: e.g. ?e Exception ?b ?e Result ?a => ?a Exception ?b
                # len ==10, body relation1 index == 1, body relation2 index == 4, head relation index == 8

                rule_var = []   # variables in current rule
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

                else:   # body size == 2
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
                original_facts = open(original_triples_path, 'r')
                facts_reader = csv.reader(original_facts, delimiter='\t')
                br1 = []
                br2 = []
                hr = []

                for row_fact in facts_reader:
                    if row_fact[1] == body_relation1:
                        br1.append(row_fact)
                    if row_fact[1] == head_relation:
                        hr.append(row_fact)
                    if row_fact[1] == body_relation2 and body_relation2:    # body size == 2
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

                        new_writer.writerow([facts, None, current_new_fact])

                else:   # body size == 2
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
                                new_writer.writerow([facts1, facts2, current_new_fact])

def generate_dict(target_dir):
    entity2id = dict()
    relation2id = dict()
    train_triples = list()
    valid_triples = list()
    test_triples = list()
    eid = 0
    rid = 0
    with open(os.path.join(target_dir, 'train.txt')) as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            train_triples.append([head, relation, tail])
            if head not in entity2id:
                entity2id[head] = eid
                eid = eid + 1
            if tail not in entity2id:
                entity2id[tail] = eid
                eid = eid + 1
            if relation not in relation2id:
                relation2id[relation] = rid
                rid = rid + 1

    with open(os.path.join(target_dir, '_aux.txt')) as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            train_triples.append([head, relation, tail])
            if head not in entity2id:
                entity2id[head] = eid
                eid = eid + 1
            if tail not in entity2id:
                entity2id[tail] = eid
                eid = eid + 1
            if relation not in relation2id:
                relation2id[relation] = rid
                rid = rid + 1

    with open(os.path.join(target_dir, 'valid.txt')) as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            valid_triples.append([head, relation, tail])
            if head not in entity2id:
                entity2id[head] = eid
                eid = eid + 1
            if tail not in entity2id:
                entity2id[tail] = eid
                eid = eid + 1
            if relation not in relation2id:
                relation2id[relation] = rid
                rid = rid + 1

    with open(os.path.join(target_dir, 'test.txt')) as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            test_triples.append([head, relation, tail])
            if head not in entity2id:
                entity2id[head] = eid
                eid = eid + 1
            if tail not in entity2id:
                entity2id[tail] = eid
                eid = eid + 1
            if relation not in relation2id:
                relation2id[relation] = rid
                rid = rid + 1

    with open(os.path.join(target_dir, 'entity2id.txt'), 'w') as f:
        for key in entity2id:
            f.writelines(key + '\t' + str(entity2id[key]))
            f.write('\n')

    with open(os.path.join(target_dir, 'relation2id.txt'), 'w') as f:
        for key in relation2id:
            f.writelines(key + '\t' + str(relation2id[key]))
            f.write('\n')

def m_n_s(txt_source_path, target_dir, minhc=0.01, minc=None, minpca=None):
    amie_arg = "java -XX:-UseGCOverheadLimit -Xmx4G -jar " + "amie_plus.jar" + " " + txt_source_path

    if minhc != 0.01 and minhc:
        amie_arg += " -minhc "
        amie_arg += str(minhc)
    if minc:
        amie_arg += " -minc "
        amie_arg += str(minc)
    if minpca:
        amie_arg += " -minpca "
        amie_arg += str(minpca)

    # tsv_target_path = "result_" + tsv_source_path
    if os.path.exists(os.path.join(target_dir, "result_pool.txt")):
        os.remove(os.path.join(target_dir, "result_pool.txt"))
    amie_arg += " >> " + os.path.join(target_dir, "result_pool.txt")
    print(amie_arg)
    os.system(amie_arg)

    cols = [[] for x in range(11)]
    with open(os.path.join(target_dir, "result_pool.txt"), 'r') as source:
        for line in source:
            if line[0] == '?':
                current_line = line.split()

                if len(current_line) != 0:
                    current_rule = ''
                    if len(current_line) == 20:
                        for i in range(0, 10):
                            current_rule = current_rule + current_line[i] + ' '
                        cols[0].append(current_rule)

                        for j in range(10, 20):
                            if current_line[j][0] != '?':
                                cols[j - 9].append(float(current_line[j]))
                            else:
                                cols[j - 9].append(current_line[j])
                    else:
                        for i in range(0, 7):
                            current_rule = current_rule + current_line[i] + ' '
                        cols[0].append(current_rule)

                        for j in range(7, 17):
                            if current_line[j][0] != '?':
                                cols[j - 6].append(float(current_line[j]))
                            else:
                                cols[j - 6].append(current_line[j])

    df = pandas.DataFrame({
            'rule': cols[0],
            'v1': cols[1],
            'v2': cols[2],
            'v3': cols[3],
            'v4': cols[4],
            'v5': cols[5],
            'v6': cols[6],
            'v7': cols[7],
            'v8': cols[8],
            'v9': cols[9],
            'v10': cols[10],
        })

    current_dir = os.path.abspath(os.path.dirname("__file__"))

    pca_sorted_path = os.path.join(target_dir, "pca_sorted_rule_pool.txt")
    std_sorted_path = os.path.join(target_dir, "std_sorted_rule_pool.txt")
    sorted_by_pca = df.sort_values(by='v3', ascending=False)
    sorted_by_std = df.sort_values(by='v2', ascending=False)
    sorted_by_pca.to_csv(pca_sorted_path, sep='\t')
    sorted_by_std.to_csv(std_sorted_path, sep='\t')

    return pca_sorted_path, std_sorted_path

if __name__ == "__main__":
    # arguments setting

    parser = argparse.ArgumentParser(description='API to handle the logic rule mining on KB via AMIE+')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-wp", "--whole_pipeline", default=True)
    group.add_argument("-m", "--mine_rule", default=False)
    group.add_argument("-p", "--predict", default=False)
    parser.add_argument("--target_dir", type=str, default="../data/")
    parser.add_argument("--triples_to_extract_rules", type=str, default="train.txt")
    parser.add_argument("--triples_to_predict", type=str, default="_aux.txt")
    parser.add_argument("--rule_path", type=str, default="rule_pool.txt")
    parser.add_argument("--new_prediction_path", type=str, default="vn_pool.txt")
    parser.add_argument("--minhc", type=float, default=0.01, help="min head coverage threshold")
    parser.add_argument("--minc", type=float, default=0.01, help="min std confidence threshold")
    parser.add_argument("--minpca", type=float, default=0.01, help="min pca confidence threshold")

    args = parser.parse_args()

    if args.whole_pipeline:
        sub_dataset = ["subject-5", "subject-10", "subject-15", "subject-20", "subject-25", "object-5", "object-10",  "object-15", "object-20", "object-25"]
        dataset = ["wn18", "wn18rr", "fb15k", "fb15k237", "yago37"]
        for dt in dataset:
            if dt is "wn18" or dt is "wn18rr":
                args.minpca = 0.7
                args.minhc = 0.3
                args.minc = 0.3
            elif dt is "fb15k" or dt is "fb15k237":
                args.minpca = 0.7
                args.minhc = 0.5
                args.minc = 0.5
            else:
                args.minpca = 0.5
                args.minhc = 0.01
                args.minc = 0.01
            dir = os.path.join(args.target_dir, dt)
            for sdt in sub_dataset:
                target_dir = os.path.join(dir, sdt)
                generate_dict(target_dir)
                sorted_rule_paths = m_n_s(txt_source_path=os.path.join(target_dir, args.triples_to_extract_rules),
                                          target_dir=target_dir, minhc=args.minhc, minc=args.minc, minpca=args.minpca)
                derive_new_triples(mined_rules_path=sorted_rule_paths[0],
                                   original_triples_path=os.path.join(target_dir, args.triples_to_predict),
                                   new_triples_path=os.path.join(target_dir, args.new_prediction_path))

    elif args.mine_rule:
        m_n_s(txt_source_path=args.triples_to_extract_rules, target_dir=args.target_dir,
              minhc=args.minhc, minc=args.minc, minpca=args.minpca)

    elif args.predict:
        derive_new_triples(mined_rules_path=args.rule_path, original_triples_path=args.triples_to_predict,
                           new_triples_path=args.new_prediction_path)
