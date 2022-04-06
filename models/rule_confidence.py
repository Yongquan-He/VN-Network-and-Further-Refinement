import torch


def get_confidence_for_one_hop_rule(triple_pre, triple_ground, model):
    score_pre = model.get_score(triple_pre, embedding)
    score_ground = model.get_score(triple_ground, embedding)
    score = score_pre - score_ground
    score = torch.sum(score, dim=1)
    score = torch.sigmoid(score)
    return score


def get_confidence_for_two_hop_rule(triple_pre1, triple_pre2, triple_ground, model):
    score_pre1 = model.get_score(triple_pre1, embedding)
    score_pre2 = model.get_score(triple_pre2, embedding)
    score_ground = model.get_score(triple_ground, embedding)
    score = score_pre1 * score_pre2 - score_ground
    score = torch.sum(score, dim=1)
    score = torch.sigmoid(score)
    return score

