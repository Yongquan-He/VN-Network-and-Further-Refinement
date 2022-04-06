import os
import random


def get_raw_train_valid_test(path):
    train_list = []
    valid_list = []
    test_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name.endswith("train.txt"):
                train_list.append(os.path.join(root, name))
            elif name.endswith("valid.txt"):
                valid_list.append(os.path.join(root, name))
            elif name.endswith("test.txt"):
                test_list.append(os.path.join(root, name))
            else:
                continue
    return train_list, valid_list, test_list


def read_triples_from_file(file_path):
    triplets = []

    with open(file_path) as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            triplets.append((head, relation, tail))
    f.close()
    return triplets


def write_triples_to_file(file_path, triples):
    with open(file_path, 'w') as f:
        for head, relation, tail in triples:
            f.write(head + '\t' + relation + '\t' + tail + '\n')
            f.flush()
    f.close()


def write_triples_to_file_batch(file_path, triples):
    ls = []
    for head, relation, tail in triples:
        ls.append(head + '\t' + relation + '\t' + tail + '\n')
    with open(file_path, 'w') as f:
        f.writelines(ls)
        f.flush()
    f.close()


def construct_train_valid_test(train_paths, valid_paths, test_paths):
    for raw_train, raw_valid, raw_test in zip(train_paths, valid_paths, test_paths):
        train_triples = read_triples_from_file(raw_train)
        valid_triples = read_triples_from_file(raw_valid)
        test_triples = read_triples_from_file(raw_test)
        sample_percentage = [25, 20, 15, 10, 5]
        for percentage in sample_percentage:
            s_train, s_aux, s_valid, s_test = get_train_valid_test_differ_percentage(train_triples, valid_triples, test_triples, percentage, True)
            folder_path = raw_train.replace("train.txt", "") + "subject-" + str(percentage)
            exist = os.path.exists(folder_path)
            if not exist:
                os.makedirs(folder_path)
            write_triples_to_file(os.path.join(folder_path, 'train.txt'), s_train)
            write_triples_to_file(os.path.join(folder_path, '_aux.txt'), s_aux)
            write_triples_to_file(os.path.join(folder_path, 'valid.txt'), s_valid)
            write_triples_to_file(os.path.join(folder_path, 'test.txt'), s_test)

            o_train, o_aux, o_valid, o_test = get_train_valid_test_differ_percentage(train_triples, valid_triples,
                                                                                     test_triples, percentage, False)
            folder_path = raw_train.replace("train.txt", "") + "object-" + str(percentage)
            exist = os.path.exists(folder_path)
            if not exist:
                os.makedirs(folder_path)
            write_triples_to_file(os.path.join(folder_path, 'train.txt'), o_train)
            write_triples_to_file(os.path.join(folder_path, '_aux.txt'), o_aux)
            write_triples_to_file(os.path.join(folder_path, 'valid.txt'), o_valid)
            write_triples_to_file(os.path.join(folder_path, 'test.txt'), o_test)


def get_train_valid_test_differ_percentage(train_triples, valid_triples, test_triples, percentage, subject):
    train = []
    valid = []
    aux = []
    test = []
    final_test = []

    candidate = set()
    observed_entity = set()
    ookb_entity = set()

    length = len(test_triples)
    sample_size = int(length * (percentage/100.0))

    random.shuffle(test_triples)
    test = test + test_triples[0:sample_size]
    for head, relation, tail in test:
        if subject:
            candidate.add(head)
        else:
            candidate.add(tail)
    for head, relation, tail in train_triples:
        if head not in candidate and tail not in candidate:
            train.append((head, relation, tail))
            observed_entity.add(head)
            observed_entity.add(tail)

        if head in candidate and tail in observed_entity:
            aux.append((head, relation, tail))
            ookb_entity.add(head)

        if head in observed_entity and tail in candidate:
            aux.append((head, relation, tail))
            ookb_entity.add(tail)

    for head, relation, tail in valid_triples:
        if head not in ookb_entity and tail not in ookb_entity:
            valid.append((head, relation, tail))

    for head, relation, tail in test:
        if (head in ookb_entity and tail in observed_entity) or (tail in ookb_entity and head in observed_entity):
            final_test.append((head, relation, tail))

    return train, aux, valid, final_test


if __name__ == "__main__":
    train_list, valid_list, test_list = get_raw_train_valid_test("../data/")
    construct_train_valid_test(train_list, valid_list, test_list)