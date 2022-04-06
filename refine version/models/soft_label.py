import torch
import torch.nn as nn


def get_soft_labels_for_one_hop(all_triples, v_triples, confidence, c, model, embedding):
    prefix = model.get_score(v_triples, embedding)
    prefix = torch.sum(prefix, dim=1)
    prefix = torch.sigmoid(prefix)

    suffix = model.get_score(all_triples, embedding)
    suffix = torch.sum(suffix, dim=1)
    suffix = torch.sigmoid(suffix)
    suffix = suffix * confidence
    suffix = suffix * c

    labels = prefix + suffix
    relu = nn.ReLU()
    labels = relu(labels)
    ones = torch.ones(len(all_triples))
    labels = torch.min(labels, ones)
    return labels

def get_soft_labels_for_two_hop(all_triples1, all_triples2, v_triples, confidence, c, model, embedding):
    prefix = model.get_score(v_triples, embedding)
    prefix = torch.sum(prefix, dim=1)
    prefix = torch.sigmoid(prefix)

    suffix = model.get_score(all_triples1, embedding) * model.get_score(all_triples2, embedding)
    suffix = torch.sum(suffix, dim=1)
    suffix = torch.sigmoid(suffix)
    suffix = suffix * confidence
    suffix = suffix * c

    labels = prefix + suffix
    relu = nn.ReLU()
    labels = relu(labels)
    ones = torch.ones(len(all_triples1))
    labels = torch.min(labels, ones)
    return labels

