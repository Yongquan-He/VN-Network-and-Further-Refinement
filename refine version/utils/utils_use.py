"""
Utility functions for link prediction
Most code is adapted from authors' implementation of RGCN link prediction:
https://github.com/MichSchli/RelationPrediction
"""
import traceback
from _thread import start_new_thread
from functools import wraps

import numpy as np
import torch
from torch.multiprocessing import Queue
import dgl
import time

import datetime
import logging
import logging.handlers
import os
import shutil
import subprocess
#######################################################################
#
# Utility function for building training and testing graphs
#
#######################################################################

def get_adj_and_degrees(num_nodes, triplets):
    """ Get adjacency list and degrees of the graph
    """
    adj_list = [[] for _ in range(num_nodes)]
    for i,triplet in enumerate(triplets):
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]
    return adj_list, degrees

def sample_edge_neighborhood(adj_list, degrees, n_triplets, sample_size):
    """Sample edges by neighborhool expansion.
    This guarantees that the sampled edges form a connected graph, which
    may help deeper GNNs that require information from more than one hop.
    """
    edges = np.zeros((sample_size), dtype=np.int32)

    #initialize
    sample_counts = np.array([d for d in degrees])
    picked = np.array([False for _ in range(n_triplets)])
    seen = np.array([False for _ in degrees])

    for i in range(0, sample_size):
        weights = sample_counts * seen

        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
            weights[np.where(sample_counts == 0)] = 0

        probabilities = (weights) / np.sum(weights)
        chosen_vertex = np.random.choice(np.arange(degrees.shape[0]),
                                         p=probabilities)
        chosen_adj_list = adj_list[chosen_vertex]
        seen[chosen_vertex] = True

        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
        chosen_edge = chosen_adj_list[chosen_edge]
        edge_number = chosen_edge[0]

        while picked[edge_number]:
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

        edges[i] = edge_number
        other_vertex = chosen_edge[1]
        picked[edge_number] = True
        sample_counts[chosen_vertex] -= 1
        sample_counts[other_vertex] -= 1
        seen[other_vertex] = True

    return edges

def sample_edge_uniform(n_triplets, sample_size):
    """Sample edges uniformly from all the edges."""
    all_edges = np.arange(n_triplets)
    return np.random.choice(all_edges, sample_size, replace=False)

def generate_sampled_graph_and_labels(triplets, sample_size, split_size,
                                      num_rels, adj_list, degrees,
                                      negative_rate, sampler="uniform"):
    """Get training graph and signals
    First perform edge neighborhood sampling on graph, then perform negative
    sampling to generate negative samples
    """
    # perform edge neighbor sampling


    if sampler == "uniform":
        edges = sample_edge_uniform(len(triplets), sample_size)
    elif sampler == "neighbor":
        edges = sample_edge_neighborhood(adj_list, degrees, len(triplets), sample_size)
    else:
        raise ValueError("Sampler type must be either 'uniform' or 'neighbor'.")

    # relabel nodes to have consecutive node ids
    edges = triplets[edges]
    src, rel, dst = edges.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_v),
                                        negative_rate)

    # Apply dropout: further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)
    src = src[graph_split_ids]
    dst = dst[graph_split_ids]
    rel = rel[graph_split_ids]

    # build DGL graph
    print("# sampled nodes: {}".format(len(uniq_v)))
    print("# sampled edges: {}".format(len(src) * 2))
    g, rel, norm = build_graph_from_triplets(len(uniq_v), num_rels,
                                             (src, rel, dst))
    return g, uniq_v, rel, norm, samples, labels

def comp_deg_norm(g):
    g = g.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm

def build_graph_from_triplets(num_nodes, num_rels, triplets):
    """ Create a DGL graph. The graph is bidirectional because RGCN authors
        use reversed relations.
        This function also generates edge type and normalization factor
        (reciprocal of node incoming degree)
    """
    g = dgl.graph(([], []))
    g.add_nodes(num_nodes)
    src, rel, dst = triplets
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))
    edges = sorted(zip(dst, src, rel))
    dst, src, rel = np.array(edges).transpose()
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    print("# nodes: {}, # edges: {}".format(num_nodes, len(src)))
    return g, rel.astype('int64'), norm.astype('int64')

def build_test_graph(num_nodes, num_rels, edges):
    src, rel, dst = edges.transpose()
    print("Test graph:")
    return build_graph_from_triplets(num_nodes, num_rels, (src, rel, dst))

def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.randint(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels

#######################################################################
#
# Utility functions for evaluations (raw)
#
#######################################################################

def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1), as_tuple=False)
    indices = indices[:, 1].view(-1)
    return indices

def perturb_and_get_raw_rank(embedding, w, a, r, b, test_size, batch_size=100):
    """ Perturb one element in the triplets
    """
    n_batch = (test_size + batch_size - 1) // batch_size
    ranks = []
    for idx in range(n_batch):
        print("batch {} / {}".format(idx, n_batch))
        batch_start = idx * batch_size
        batch_end = min(test_size, (idx + 1) * batch_size)
        batch_a = a[batch_start: batch_end]
        batch_r = r[batch_start: batch_end]
        emb_ar = embedding[batch_a] * w[batch_r]
        emb_ar = emb_ar.transpose(0, 1).unsqueeze(2) # size: D x E x 1
        emb_c = embedding.transpose(0, 1).unsqueeze(1) # size: D x 1 x V
        # out-prod and reduce sum
        out_prod = torch.bmm(emb_ar, emb_c) # size D x E x V
        score = torch.sum(out_prod, dim=0) # size E x V
        score = torch.sigmoid(score)
        target = b[batch_start: batch_end]
        ranks.append(sort_and_rank(score, target))
    return torch.cat(ranks)

# return MRR (raw), and Hits @ (1, 3, 10)
def calc_raw_mrr(embedding, w, test_triplets, hits=[], eval_bz=100):
    with torch.no_grad():
        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        o = test_triplets[:, 2]
        test_size = test_triplets.shape[0]

        # perturb subject
        ranks_s = perturb_and_get_raw_rank(embedding, w, o, r, s, test_size, eval_bz)
        # perturb object
        ranks_o = perturb_and_get_raw_rank(embedding, w, s, r, o, test_size, eval_bz)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())
        print("MRR (raw): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))
    return mrr.item()

###########################  checkpoint = torch.load(pretrained_model_state_file)############################################
#
# Utility functions for evaluations (filtered)
#
#######################################################################

def filter_o(triplets_to_filter, target_s, target_r, target_o, num_entities):
    target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
    filtered_o = []
    # Do not filter out the test triplet, since we want to predict on it
    if (target_s, target_r, target_o) in triplets_to_filter:
        triplets_to_filter.remove((target_s, target_r, target_o))
    # Do not consider an object if it is part of a triplet to filter
    for o in range(num_entities):
        if (target_s, target_r, o) not in triplets_to_filter:
            filtered_o.append(o)
    return torch.LongTensor(filtered_o)

def filter_s(triplets_to_filter, target_s, target_r, target_o, num_entities):
    target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
    filtered_s = []
    # Do not filter out the test triplet, since we want to predict on it
    if (target_s, target_r, target_o) in triplets_to_filter:
        triplets_to_filter.remove((target_s, target_r, target_o))
    # Do not consider a subject if it is part of a triplet to filter
    for s in range(num_entities):
        if (s, target_r, target_o) not in triplets_to_filter:
            filtered_s.append(s)
    return torch.LongTensor(filtered_s)

def perturb_o_and_get_filtered_rank(embedding, w, s, r, o, test_size, triplets_to_filter):
    """ Perturb object in the triplets
    """
    num_entities = embedding.shape[0]
    ranks = []
    for idx in range(test_size):
        if idx % 100 == 0:
            print("test triplet {} / {}".format(idx, test_size))
        target_s = s[idx]
        target_r = r[idx]
        target_o = o[idx]
        filtered_o = filter_o(triplets_to_filter, target_s, target_r, target_o, num_entities)
        target_o_idx = int((filtered_o == target_o).nonzero())
        emb_s = embedding[target_s]
        emb_r = w[target_r]
        emb_o = embedding[filtered_o]
        emb_triplet = emb_s * emb_r * emb_o
        scores = torch.sigmoid(torch.sum(emb_triplet, dim=1))
        _, indices = torch.sort(scores, descending=True)
        rank = int((indices == target_o_idx).nonzero())
        ranks.append(rank)
    return torch.LongTensor(ranks)

def perturb_s_and_get_filtered_rank(embedding, w, s, r, o, test_size, triplets_to_filter):
    """ Perturb subject in the triplets
    """
    num_entities = embedding.shape[0]
    ranks = []
    for idx in range(test_size):
        if idx % 100 == 0:
            print("test triplet {} / {}".format(idx, test_size))
        target_s = s[idx]
        target_r = r[idx]
        target_o = o[idx]
        filtered_s = filter_s(triplets_to_filter, target_s, target_r, target_o, num_entities)
        target_s_idx = int((filtered_s == target_s).nonzero())
        emb_s = embedding[filtered_s]
        emb_r = w[target_r]
        emb_o = embedding[target_o]
        emb_triplet = emb_s * emb_r * emb_o
        scores = torch.sigmoid(torch.sum(emb_triplet, dim=1))
        _, indices = torch.sort(scores, descending=True)
        rank = int((indices == target_s_idx).nonzero())
        # print("score rank: ")
        # print(_)
        # print("the rank of target entity: %d" % rank)
        # print("the score of target entity: %f" % _[rank])
        ranks.append(rank)
    return torch.LongTensor(ranks)

def calc_filtered_mrr(embedding, w, train_triplets, valid_triplets, test_triplets, hits=[]):
    with torch.no_grad():
        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        o = test_triplets[:, 2]
        test_size = test_triplets.shape[0]

        triplets_to_filter = torch.cat([train_triplets, valid_triplets, test_triplets]).tolist()
        triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter}
        print('Perturbing subject...')
        ranks_s = perturb_s_and_get_filtered_rank(embedding, w, s, r, o, test_size, triplets_to_filter)
        print('Perturbing object...')
        ranks_o = perturb_o_and_get_filtered_rank(embedding, w, s, r, o, test_size, triplets_to_filter)

        ranks = torch.cat([ranks_s, ranks_o])
        # ranks = ranks_o
        ranks += 1 # change to 1-indexed
        ranks_o += 1
        ranks_s += 1
        mr_s = torch.mean(ranks_s.float())
        mr_o = torch.mean(ranks_o.float())
        mr = torch.mean(ranks.float())
        mrr_s = torch.mean(1.0 / ranks_s.float())
        mrr_o = torch.mean(1.0 / ranks_o.float())
        mrr = torch.mean(1.0 / ranks.float())
        print("MR_S (filtered): {:.6f}".format(mr_s.item()))
        print("MR_O (filtered): {:.6f}".format(mr_o.item()))
        print("MR (filtered): {:.6f}".format(mr.item()))
        print("MRR_S (filtered): {:.6f}".format(mrr_s.item()))
        print("MRR_O (filtered): {:.6f}".format(mrr_o.item()))
        print("MRR (filtered): {:.6f}".format(mrr.item()))
        hits_score_s = []
        hits_score_o = []
        hits_score = []
        for hit in hits:
            avg_count_s = torch.mean((ranks_s <= hit).float())
            print("Subject: Hits (filtered) @ {}: {:.6f}".format(hit, avg_count_s.item()))
            hits_score_s.append(avg_count_s.item())
        for hit in hits:
            avg_count_o = torch.mean((ranks_o <= hit).float())
            print("Object: Hits (filtered) @ {}: {:.6f}".format(hit, avg_count_o.item()))
            hits_score_o.append(avg_count_o.item())
        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            print("Hits (filtered) @ {}: {:.6f}".format(hit, avg_count.item()))
            hits_score.append(avg_count.item())

    return mrr_s.item(), mrr_o.item(), mrr.item(), hits_score_s, hits_score_o, hits_score

#######################################################################
#
# Main evaluation function
#
#######################################################################

def calc_mrr(embedding, w, train_triplets, valid_triplets, test_triplets, hits=[], eval_bz=100, eval_p="filtered"):
    if eval_p == "filtered":
        mrr_s, mrr_o, mrr, hits_s, hits_o, hits = calc_filtered_mrr(embedding, w, train_triplets, valid_triplets, test_triplets, hits)
    else:
        mrr = calc_raw_mrr(embedding, w, test_triplets, hits, eval_bz)
    return mrr_s, mrr_o, mrr, hits_s, hits_o, hits


#######################################################################
#
# Multithread wrapper
#
#######################################################################

# According to https://github.com/pytorch/pytorch/issues/17199, this decorator
# is necessary to make fork() and openmp work together.
def thread_wrapped_func(func):
    """
    Wraps a process entry point to make it work with OpenMP.
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = Queue()
        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)
    return decorated_function

#######################################################################
#
# train step
#
#######################################################################
def node_norm_to_edge_norm(g, node_norm):
    g = g.local_var()
    # convert to edge norm
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
    return g.edata['norm']


def train_step(model, args, train_data, num_rels, adj_list, degrees,
               optimizer):
    model.train()

    # perform edge neighborhood sampling to generate training graph and data
    # return g, uniq_v, rel, norm, samples, labels

    g, node_id, edge_type, node_norm, data, labels = \
        generate_sampled_graph_and_labels(
            train_data, args.graph_batch_size, args.graph_split_size,
            num_rels, adj_list, degrees, args.negative_sample,
            args.edge_sampler)
    print("Done edge sampling")

    # set node/edge feature
    node_id = torch.from_numpy(node_id).view(-1, 1).long()
    edge_type = torch.from_numpy(edge_type)
    edge_norm = node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
    data, labels = torch.from_numpy(data), torch.from_numpy(labels)
    deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)

    node_id, deg = node_id.cuda(), deg.cuda()
    edge_type, edge_norm = edge_type.cuda(), edge_norm.cuda()
    data, labels = data.cuda(), labels.cuda()
    g = g.to(args.gpu)

    # start to train the model
    t0 = time.time()
    embed = model(g, node_id, edge_type, edge_norm)
    loss = model.get_loss(embed, data, labels)
    t1 = time.time()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
    optimizer.step()
    t2 = time.time()

    forward_time = t1 - t0
    backward_time = t2 - t1

    # Release the GPU
    del g, node_id, edge_type, edge_norm, data, labels
    torch.cuda.empty_cache()
    optimizer.zero_grad()

    return loss.item(), forward_time, backward_time

def batch_train_step(ts, logger, epoch, best_mrr, model, args, data, num_rels, optimizer, adj_list=None, degrees=None):
    model.train()
    batch = 0
    # perform edge neighborhood sampling to generate training graph and data
    # return g, uniq_v, rel, norm, samples, labels
    for batch_data in batch_by_size_no_labels(args.graph_batch_size, data):
        batch += 1
        # relabel nodes to have consecutive node ids
        edges = batch_data
        src, rel, dst = edges.transpose()
        uniq_v, edges = np.unique((src, dst), return_inverse=True)
        src, dst = np.reshape(edges, (2, -1))
        relabeled_edges = np.stack((src, rel, dst)).transpose()

        # negative sampling
        samples, labels = negative_sampling(relabeled_edges, len(uniq_v),
                                            args.negative_sample)

        # build DGL graph
        print("# sampled nodes: {}".format(len(uniq_v)))
        print("# sampled edges: {}".format(len(src) * 2))
        g, rel, norm = build_graph_from_triplets(len(uniq_v), num_rels,
                                                 (src, rel, dst))
        node_id = uniq_v
        edge_type = rel
        node_norm = norm
        data = samples

        # set node/edge feature
        node_id = torch.from_numpy(node_id).view(-1, 1).long()
        edge_type = torch.from_numpy(edge_type)
        edge_norm = node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
        data, labels = torch.from_numpy(data), torch.from_numpy(labels)
        deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)

        node_id, deg = node_id.cuda(), deg.cuda()
        edge_type, edge_norm = edge_type.cuda(), edge_norm.cuda()
        data, labels = data.cuda(), labels.cuda()
        g = g.to(args.gpu)

        # start to train the model
        t0 = time.time()

        embed = model(g, node_id, edge_type, edge_norm)
        loss = model.get_loss(embed, data, labels)

        gpu_usage = get_gpu_memory_map()[int(args.gpu)]
        logger.info('GPU usage for embeddings %d' % gpu_usage)

        t1 = time.time()
        loss.backward()

        gpu_usage = get_gpu_memory_map()[int(args.gpu)]
        logger.info('GPU usage for backward %d' % gpu_usage)

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
        optimizer.step()
        t2 = time.time()

        forward_time = t1 - t0
        backward_time = t2 - t1

        # # Release the GPU
        # del g, node_id, edge_type, edge_norm, data, labels
        # torch.cuda.empty_cache()
        optimizer.zero_grad()

        print("Training for aux data: Epoch {:04d} | Batch {:04d} | Loss {:.4f} | Best MRR {:.4f} | Forward {:.4f}s | Backward {:.4f}s".
              format(epoch, batch, loss.item(), best_mrr, forward_time, backward_time))
        logger.info("Training for aux data: Epoch {:04d} | Batch {:04d} | Loss {:.4f} | Best MRR {:.4f} | [{:.1f} s]".
                    format(epoch, batch, loss.item(), best_mrr, time.time() - ts))

def batch_by_num(n_batch, data, soft_labels):
    n = np.size(data, 0)
    if n < n_batch:
        ret = data, soft_labels
        yield ret
    else:
        for i in range(n):
                head = int(n*i/n_batch)
                tail = int(n*(i+1)/n_batch)
                ret = data[head:tail, :], soft_labels[head:tail]
                yield ret

def batch_by_size(batch_size, data, soft_labels):
    n = np.size(data, 0)
    batch_num = int(n / batch_size)

    if n < batch_size:
        yield data, soft_labels, False
    else:
        for i in range(batch_num):
            head = int(i*batch_size)
            tail = int((i+1)*batch_size)
            ret = data[head:tail, :], soft_labels[head:tail], False
            yield ret
        ret = data[tail:, ], soft_labels[tail:], True
        yield ret

def batch_by_size_no_labels(batch_size, data, aux_data=None):
    if aux_data is None:
        # void changing the original data when shuffling
        data = data.copy()
        np.random.shuffle(data)
        n = np.size(data, 0)
        batch_num = int(n / batch_size)
        batch_num = np.max((1, batch_num))
        if n < batch_size:
            yield data
        else:
            for i in range(batch_num):
                head = int(i*batch_size)
                tail = int((i+1)*batch_size)
                ret = data[head:tail, :]
                yield ret
            ret = data[tail:, ]
            yield ret
    else:
        # void changing the original data when shuffling
        data = data.copy()
        aux_data = aux_data.copy()
        np.random.shuffle(data)
        np.random.shuffle(aux_data)
        n = np.size(data, 0)
        n_aux = np.size(aux_data, 0)
        batch_num = int(n / batch_size)
        batch_num = np.max((1, batch_num))
        batch_size_aux = int(n_aux / batch_num)
        if n < batch_size:
            yield data, batch_size_aux[:4*n]
        else:
            for i in range(batch_num):
                head = int(i * batch_size)
                tail = int((i + 1) * batch_size)
                head_aux = int(i * batch_size_aux)
                tail_aux = int((i + 1) * batch_size_aux)
                ret = data[head:tail, :], aux_data[head_aux:tail_aux, :]
                yield ret
            ret = data[tail:], aux_data[tail_aux:]
            yield ret

def batch_by_size_softlabels(batch_size, data, aux_data, soft_labels):
    # void changing the original data when shuffling
    # data = data.copy()
    # aux_data = aux_data.copy()
    # np.random.shuffle(data)
    # np.random.shuffle(aux_data)
    assert data.shape[0] == soft_labels.shape[0]
    p = np.random.permutation(data.shape[0])
    data, soft_labels = data[p], soft_labels[p]

    p = np.random.permutation(aux_data.shape[0])
    aux_data = aux_data[p]

    n = np.size(data, 0)
    n_aux = np.size(aux_data, 0)

    batch_num = int(n_aux / batch_size)
    batch_num = np.max((1, batch_num))
    batch_size_virtual = int(n / batch_num)
    if n < batch_size:
        yield soft_labels, data, aux_data,
    else:
        for i in range(batch_num):
            head = int(i * batch_size_virtual)
            tail = int((i + 1) * batch_size_virtual)
            head_aux = int(i * batch_size)
            tail_aux = int((i + 1) * batch_size)
            ret = soft_labels[head:tail], data[head:tail, :], aux_data[head_aux:tail_aux, :]
            yield ret
        ret = soft_labels[tail:], data[tail:], aux_data[tail_aux:]
        yield ret
# def batch_by_size_v_p_a(batch_size, data, premise, aux_data=None):

def negative_sampling_for_virtual_data(pos_samples, num_entity, negative_rate, total_samples=None):
    ts = time.time()
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate), dtype=np.float32)
    values = np.random.randint(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    # total_samples = [list(sample) for sample in total_samples]
    # neg_samples = [sample for sample in neg_samples if list(sample) not in total_samples]
    # len_neg = len(neg_samples)
    #
    # neg_samples = np.asarray(neg_samples)
    # labels = np.zeros(len_neg, dtype=np.float32)
    # print("negative samples number:", len_neg)

    print("sampling time cost:", time.time() - ts)
    return np.concatenate((pos_samples, neg_samples)), labels

# def calculate_sigma(model, embeddings, probs, premises):

def relabel_nodes(nodes, uniq_v):
    def relabel_node(node):
        return np.where(uniq_v == node)[0][0]
    return np.asarray(list(map(relabel_node, nodes)))

def relabel_triples(triples, uniq_v):
    src, rel, dst = triples.transpose()
    src = relabel_nodes(src, uniq_v)
    dst = relabel_nodes(dst, uniq_v)
    relabeled_triples = np.stack((src, rel, dst)).transpose()
    return relabeled_triples


def timestamp():
    now = datetime.datetime.now()
    now_str = now.strftime("%Y%m%d%H%M")
    return now_str

def get_logger(logdir, logname, loglevel=logging.INFO):
    fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    formatter = logging.Formatter(fmt)

    handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(logdir, logname),
        maxBytes=2 * 1024 * 1024 * 1024,
        backupCount=10)
    handler.setFormatter(formatter)

    logger = logging.getLogger("")
    logger.addHandler(handler)
    logger.setLevel(loglevel)
    return logger

def makedirs(dir, force=False):
    if os.path.exists(dir):
        if force:
            shutil.rmtree(dir)
            os.makedirs(dir)
    else:
        os.makedirs(dir)

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def load_vn_data(file_path, triples_set, another_set):
    print("load one hop and two hop data from {}".format(file_path))

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
    n_cnt = 0
    file_path_vn = os.path.join(file_path, 'vn.txt')
    with open(file_path_vn) as f:
        for line in f:
            grounding_list = line.strip().split('\t')
            grounding = eval(grounding_list[2])
            grounding_add = (entity2id[grounding[0]], relation2id[grounding[1]], entity2id[grounding[2]])
            if grounding_add in another_set:
                n_cnt = n_cnt + 1
                another_set.add(grounding_add)
            if grounding_add not in triples_set:
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
    print("Triples not in original datasets: " + str(cnt))
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
    with open(file_path_sys) as f:
        for line in f:
            list_line = eval(line)
            pre = list_line[0][0]
            ground = list_line[1][0]
            path = list_line[2][0]
            if pre[0] is 1:
                l, s1, r1, o1, s2, r2, o2, loc = pre
                pre1 = (entity2id[s1], relation2id[r1], entity2id[o1])
                pre2 = (entity2id[s2], relation2id[r2], entity2id[o2])
                l, ps1, pr1, po1, ps2, pr2, po2, loc = ground
                ground1 = (entity2id[ps1], relation2id[pr1], entity2id[po1])
                ground2 = (entity2id[ps2], relation2id[pr2], entity2id[po2])
                g_list = [ground1, ground2]
                if g_list[loc - 1] not in triples_set:
                    cnt = cnt + 1
                    if g_list[loc - 1] in another_set:
                        n_cnt = n_cnt + 1
                sys_one_pre1.append(pre1)
                sys_one_pre2.append(pre2)
                sys_one_ground1.append(ground1)
                sys_one_ground2.append(ground2)
                sys_one_confidence.append(path[len(path) - 1])
                sys_one_loc.append(loc)
            else:
                l, s1, r1, o1, s2, r2, o2, s3, r3, o3, loc = pre
                pre1 = (entity2id[s1], relation2id[r1], entity2id[o1])
                pre2 = (entity2id[s2], relation2id[r2], entity2id[o2])
                pre3 = (entity2id[s3], relation2id[r3], entity2id[o3])
                l, ps1, pr1, po1, ps2, pr2, po2, ps3, pr3, po3, loc = ground
                ground1 = (entity2id[ps1], relation2id[pr1], entity2id[po1])
                ground2 = (entity2id[ps2], relation2id[pr2], entity2id[po2])
                ground3 = (entity2id[ps3], relation2id[pr3], entity2id[po3])
                g_list = [ground1, ground2, ground3]
                if g_list[loc-1] not in triples_set:
                    cnt = cnt + 1
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

    print("Triples not in original datasets: " + str(cnt))
    print("Triples in test dataset: " + str(n_cnt))
    return one_hop_pre, one_hop_ground, two_hop_pre1, two_hop_pre2, two_hop_ground

def load_data(file_path):
    '''
        argument:
            file_path: ./data/FB15k-237

        return:
            entity2id, relation2id, train_triplets, valid_triplets, test_triplets
    '''

    print("load data from {}".format(file_path))

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

    print('num_entity: {}'.format(len(entity2id)))
    print('num_relation: {}'.format(len(relation2id)))
    print('num_train_triples: {}'.format(len(train_triplets)))
    print('num_aux_triples: {}'.format(len(aux_triplets)))
    print('num_valid_triples: {}'.format(len(valid_triplets)))
    print('num_test_triples: {}'.format(len(test_triplets)))

    return entity2id, relation2id, train_triplets, valid_triplets, aux_triplets, test_triplets

def read_triplets(file_path, entity2id, relation2id):
    triplets = []

    with open(file_path) as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            triplets.append((entity2id[head], relation2id[relation], entity2id[tail]))

    return triplets


def check_rule(ground, pre1, pre2=None):

    # one order rules
    if pre2 is None:
        if pre1[0] == ground[0] and pre1[2] == ground[2]:
            return ("eq", ground[1], pre1[1])
        elif pre1[0] == ground[2] and pre1[2] == ground[0]:
            return ("inv", ground[1], pre1[1])

    # two order rules
    else:
        pre1, pre2 = list(pre1), list(pre2)
        premise = pre1 + pre2
        a, b, cls_rel = ground[0], ground[2], ground[1]
        a_index = premise.index(a)
        b_index = premise.index(b)


        if a_index % 3 == 2 and b_index % 3 == 2:
            a_rel = int(premise[a_index - 1])
            b_rel = int(premise[b_index - 1])
            return ('chain1', cls_rel, a_rel, b_rel)

        elif a_index % 3 == 2 and b_index % 3 == 0:
            a_rel = int(premise[a_index - 1])
            b_rel = int(premise[b_index + 1])
            return ('chain2', cls_rel, a_rel, b_rel)
 
        elif a_index % 3 == 0 and b_index % 3 == 2:
            a_rel = int(premise[a_index + 1])
            b_rel = int(premise[b_index - 1])
            return ('chain3', cls_rel, a_rel, b_rel)

        elif a_index % 3 == 0 and b_index % 3 == 0:
            a_rel = int(premise[a_index + 1])
            b_rel = int(premise[b_index + 1])
            return ('chain4', cls_rel, a_rel, b_rel)
