import csv


def derive_new_triples(mined_rules_path, original_triples_path, new_triples_path):
    with open(new_triples_path, 'w', newline='', encoding='utf-8') as target:
        new_writer = csv.writer(target, delimiter='\t')
        with open(mined_rules_path, 'r', encoding='utf-8') as rules:
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