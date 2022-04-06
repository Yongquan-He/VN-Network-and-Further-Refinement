import torch
import torch.nn as nn

class RuleConfidence():
    def __init__(self, args, rel_embeddings,
                 one_hop_ground_rule,
                 one_hop_ground_rule_set,
                 two_hop_ground_rule,
                 two_hop_ground_rule_set):
        self.dim = args.embedding_dim
        self.rel_embeddings = rel_embeddings
        self.one_hop_ground_rule = one_hop_ground_rule
        self.one_hop_ground_rule_set = one_hop_ground_rule_set
        self.two_hop_ground_rule = two_hop_ground_rule
        self.two_hop_ground_rule_set = two_hop_ground_rule_set

    def get_confidence_for_one_hop_rule(self):
        rule2confidence = {}
        eq_rule = []
        inv_rule = []
        for rule in self.one_hop_ground_rule_set:
            if rule[0] == "eq":
                eq_rule.append(rule[1:])
            elif rule[0] == "inv":
                inv_rule.append(rule[1:])

        identity = torch.cat((torch.ones(int(self.dim - self.dim / 4)), torch.zeros(int(self.dim / 4))), 0).unsqueeze(0).cuda()

        if len(eq_rule) != 0:
            eq_embeddings = self.rel_embeddings[torch.LongTensor(eq_rule)]
            equivalent_prob = self.sim(head=eq_embeddings[:, 0, :], tail=eq_embeddings[:, 1, :], arity=1)
            for prob, rule in zip(equivalent_prob, eq_rule):
                rule2confidence[('eq', rule[0], rule[1])] = prob


        if len(inv_rule) != 0:
            inv_embeddings = self.rel_embeddings[torch.LongTensor(inv_rule)]
            inverse_prob = (self.sim(head=[inv_embeddings[:, 0, :], inv_embeddings[:, 1, :]],
                                tail=identity, arity=2) +
                            self.sim(head=[inv_embeddings[:, 1, :], inv_embeddings[:, 0, :]],
                                tail=identity, arity=2)) / 2
            for prob, rule in zip(inverse_prob, inv_rule):
                rule2confidence[('inv', rule[0], rule[1])] = prob


        one_hop_confidence = []
        for rule in self.one_hop_ground_rule:
            one_hop_confidence.append(rule2confidence[rule])

        return one_hop_confidence

    def get_confidence_for_two_hop_rule(self):
        rule2confidence = {}
        chain1_rule = []
        chain2_rule = []
        chain3_rule = []
        chain4_rule = []
        for rule in self.two_hop_ground_rule_set:
            if rule[0] == "chain1":
                chain1_rule.append(rule[1:])
            if rule[0] == "chain2":
                chain2_rule.append(rule[1:])
            if rule[0] == "chain3":
                chain3_rule.append(rule[1:])
            if rule[0] == "chain4":
                chain4_rule.append(rule[1:])

        identity = torch.cat((torch.ones(int(self.dim - self.dim / 4)), torch.zeros(int(self.dim / 4))), 0).unsqueeze(0).cuda()

        if len(chain1_rule) != 0:
            chain1_embeddings = self.rel_embeddings[torch.LongTensor(chain1_rule)]
            chain1_prob = self.sim(
                head=[chain1_embeddings[:, 1, :], chain1_embeddings[:, 0, :]],
                tail=chain1_embeddings[:, 2, :], arity=2)
            for prob, rule in zip(chain1_prob, chain1_rule):
                rule2confidence[('chain1', rule[0], rule[1], rule[2])] = prob

        if len(chain2_rule) != 0:
            chain2_embeddings = self.rel_embeddings[torch.LongTensor(chain2_rule)]
            chain2_prob = self.sim(
                head=[chain2_embeddings[:, 2, :], chain2_embeddings[:, 1, :], chain2_embeddings[:, 0, :]],
                tail=identity, arity=3)
            for prob, rule in zip(chain2_prob, chain2_rule):
                rule2confidence[('chain2', rule[0], rule[1], rule[2])] = prob

        if len(chain3_rule) != 0:
            chain3_embeddings = self.rel_embeddings[torch.LongTensor(chain3_rule)]
            chain3_prob = self.sim(
                head=[chain3_embeddings[:, 1, :], chain3_embeddings[:, 2, :]],
                tail=chain3_embeddings[:, 0, :], arity=2)
            for prob, rule in zip(chain3_prob, chain3_rule):
                rule2confidence[('chain3', rule[0], rule[1], rule[2])] = prob

        if len(chain4_rule) != 0:
            chain4_embeddings = self.rel_embeddings[torch.LongTensor(chain4_rule)]
            chain4_prob = self.sim(
                head=[chain4_embeddings[:, 0, :], chain4_embeddings[:, 2, :]],
                tail=chain4_embeddings[:, 1, :], arity=2)
            for prob, rule in zip(chain4_prob, chain4_rule):
                rule2confidence[('chain4', rule[0], rule[1], rule[2])] = prob
        two_hop_confidence = []
        for rule in self.two_hop_ground_rule:
            two_hop_confidence.append(rule2confidence[rule])

        return two_hop_confidence

    def sim(self, head=None, tail=None, arity=None):
        if arity == 1:
            A_scalar, A_x, A_y = self.split_embedding(head)
        elif arity == 2:
            M1_scalar, M1_x, M1_y = self.split_embedding(head[0])
            M2_scalar, M2_x, M2_y = self.split_embedding(head[1])
            A_scalar = M1_scalar * M2_scalar
            A_x = M1_x * M2_x - M1_y * M2_y
            A_y = M1_x * M2_y + M1_y * M2_x
        elif arity == 3:
            M1_scalar, M1_x, M1_y = self.split_embedding(head[0])
            M2_scalar, M2_x, M2_y = self.split_embedding(head[1])
            M3_scalar, M3_x, M3_y = self.split_embedding(head[2])
            M1M2_scalar = M1_scalar * M2_scalar
            M1M2_x = M1_x * M2_x - M1_y * M2_y
            M1M2_y = M1_x * M2_y + M1_y * M2_x
            A_scalar = M1M2_scalar * M3_scalar
            A_x = M1M2_x * M3_x - M1M2_y * M3_y
            A_y = M1M2_x * M3_y + M1M2_y * M3_x
        else:
            raise NotImplemented
        B_scala, B_x, B_y = self.split_embedding(tail)

        similarity = torch.cat(((A_scalar - B_scala) ** 2, (A_x - B_x) ** 2, (A_x - B_x) ** 2, (A_y - B_y) ** 2, (A_y - B_y) ** 2), axis=1)
        similarity = torch.sqrt(torch.sum(similarity, dim=1))

        # recale the probability
        # avoid of zero of denominator
        if torch.max(similarity) == torch.min(similarity):
            probability = similarity
        else:
            probability = (torch.max(similarity) - similarity) / (torch.max(similarity) - torch.min(similarity))
        # print(probability)
        # print("\n")

        return probability

    def split_embedding(self, embedding):
        # embedding: [None, dim]
        assert self.dim % 4 == 0
        num_scalar = self.dim // 2
        num_block = self.dim // 4
        embedding_scalar = embedding[:, 0:num_scalar]
        embedding_x = embedding[:, num_scalar:-num_block]
        embedding_y = embedding[:, -num_block:]
        return embedding_scalar, embedding_x, embedding_y


def get_soft_labels_for_one_hop(pre_triples, ground_triples, confidence, c, model, embedding, args):

    suffix = model.calc_score(embedding, pre_triples)
    prefix = model.calc_score(embedding, ground_triples)
    if not args.iter:
        suffix = suffix * confidence
        suffix = suffix * c
        labels = prefix + suffix
    else:
        labels = suffix * prefix - suffix + 1
    relu = nn.ReLU()
    labels = relu(labels)
    ones = torch.ones(len(pre_triples)).cuda()
    labels = torch.min(labels, ones)
    return labels

def get_soft_labels_for_two_hop(pre1_triples, pre2_triples, ground_triples, confidence, c, model, embedding, args):

    suffix = model.calc_score(embedding, pre1_triples) * model.calc_score(embedding, pre2_triples)
    prefix = model.calc_score(embedding, ground_triples)
    if not args.iter:
        suffix = suffix * confidence
        suffix = suffix * c
        labels = prefix + suffix
    else:
        labels = suffix * prefix - suffix + 1
    relu = nn.ReLU()
    labels = relu(labels)
    ones = torch.ones(len(pre1_triples)).cuda()
    labels = torch.min(labels, ones)
    return labels

