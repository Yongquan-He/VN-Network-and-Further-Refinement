import argparse
import os

from mine_and_sort import m_n_s
from predict import derive_new_triples

def generate_dict(target_dir):
    entity2id = dict()
    relation2id = dict()
    train_triples = list()
    valid_triples = list()
    test_triples = list()
    eid = 0
    rid = 0
    with open(os.path.join(target_dir, 'train.txt'), encoding='utf-8') as f:
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

    with open(os.path.join(target_dir, '_aux.txt'), encoding='utf-8') as f:
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

    with open(os.path.join(target_dir, 'valid.txt'), encoding='utf-8') as f:
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

    with open(os.path.join(target_dir, 'test.txt'), encoding='utf-8') as f:
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

    with open(os.path.join(target_dir, 'entity2id.txt'), 'w', encoding='utf-8') as f:
        for key in entity2id:
            f.writelines(key + '\t' + str(entity2id[key]))
            f.write('\n')

    with open(os.path.join(target_dir, 'relation2id.txt'), 'w', encoding='utf-8') as f:
        for key in relation2id:
            f.writelines(key + '\t' + str(relation2id[key]))
            f.write('\n')

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
    parser.add_argument("--rule_path", type=str, default="rule.txt")
    parser.add_argument("--new_prediction_path", type=str, default="vn.txt")
    parser.add_argument("--minhc", type=float, default=0.01, help="min head coverage threshold, default=0.01(yago37) 0.75(others)")
    parser.add_argument("--minc", type=float, default=0.01, help="min std confidence threshold, default=0.01(yago37) 0.75(others)")
    parser.add_argument("--minpca", type=float, default=0.75, help="min pca confidence threshold, default=0.75")

    args = parser.parse_args()

    if args.whole_pipeline:
        sub_dataset = ["subject-5", "subject-10", "subject-15", "subject-20", "subject-25", "object-5", "object-10",  "object-15", "object-20", "object-25"]
        dataset = ["wn18", "wn18rr", "fb15k", "fb15k237", "yago37"]
        for dt in dataset:
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
