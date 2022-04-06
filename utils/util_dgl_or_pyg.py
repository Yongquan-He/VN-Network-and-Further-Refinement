import os
import math

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_scatter import scatter_add
from torch_geometric.data import Data

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

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
    print('num_valid_triples: {}'.format(len(valid_triplets)))
    print('num_aux_triples: {}'.format(len(aux_triplets)))
    print('num_test_triples: {}'.format(len(test_triplets)))

    return entity2id, relation2id, train_triplets, valid_triplets, aux_triplets, test_triplets


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
    file_path = os.path.join(file_path, 'vn.txt')
    with open(file_path) as f:
        for line in f:
            grounding_list = line.strip().split('\t')
            grounding = eval(grounding_list[2])
            grounding_add = (entity2id[grounding[0]], relation2id[grounding[1]], entity2id[grounding[2]])
            if grounding_add in another_set:
                n_cnt = n_cnt + 1
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
    return one_hop_pre, one_hop_ground, two_hop_pre1, two_hop_pre2, two_hop_ground


def load_sys_data(file_path):
    print("load sys data from {}".format(file_path))

    with open(os.path.join(file_path, 'entities.dict')) as f:
        entity2id = dict()

        for line in f:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(file_path, 'relations.dict')) as f:
        relation2id = dict()

        for line in f:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    sys11 = []
    sys12 = []
    sys13 = []
    sys21 = []
    sys22 = []
    sys23 = []
    rel = []
    file_path = os.path.join(file_path, 'sys.txt')
    with open(file_path) as f:
        for line in f:
            s1, r1, o1, s2, r2, o2, s3, r3, o3, s4, r4, o4, s5, r5, o5, s6, r6, o6, r = line.strip().split('\t')
            sys11.append((entity2id[s1], relation2id[r1], entity2id[o1]))
            sys12.append((entity2id[s2], relation2id[r2], entity2id[o2]))
            sys13.append((entity2id[s3], relation2id[r3], entity2id[o3]))
            sys21.append((entity2id[s4], relation2id[r4], entity2id[o4]))
            sys22.append((entity2id[s5], relation2id[r5], entity2id[o5]))
            sys23.append((entity2id[s6], relation2id[r6], entity2id[o6]))
            rel.append(relation2id[r])

    return sys11, sys12, sys13, sys21, sys22, sys23, rel

def read_triplets(file_path, entity2id, relation2id):
    triplets = []

    with open(file_path) as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            triplets.append((entity2id[head], relation2id[relation], entity2id[tail]))

    return triplets

def sample_edge_uniform(n_triples, sample_size):
    """Sample edges uniformly from all the edges."""
    all_edges = np.arange(n_triples)
    return np.random.choice(all_edges, sample_size, replace=False)


def edge_normalization(edge_type, edge_index, num_entity, num_relation):
    '''
        Edge normalization trick
        - one_hot: (num_edge, num_relation)
        - deg: (num_node, num_relation)
        - index: (num_edge)
        - deg[edge_index[0]]: (num_edge, num_relation)
        - edge_norm: (num_edge)
    '''
    one_hot = F.one_hot(edge_type, num_classes=2 * num_relation).to(torch.float)
    deg = scatter_add(one_hot, edge_index[0], dim=0, dim_size = num_entity)
    index = edge_type + torch.arange(len(edge_index[0])) * (2 * num_relation)
    edge_norm = 1 / deg[edge_index[0]].view(-1)[index]

    return edge_norm

def negative_sampling(known_labels, pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = known_labels
    values = np.random.choice(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels

def generate_sampled_graph_and_labels(known_labels, sampled_triplets, split_size, num_rels, negative_rate):
    # Select sampled edges
    edges = sampled_triplets
    sample_size = len(edges)
    src, rel, dst = edges.transpose()
    uniq_entity, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # Negative sampling
    samples, labels = negative_sampling(known_labels, relabeled_edges, len(uniq_entity), negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = sample_size * split_size
    split_size = int(split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)

    src = torch.tensor(src[graph_split_ids], dtype=torch.long).contiguous()
    dst = torch.tensor(dst[graph_split_ids], dtype=torch.long).contiguous()
    rel = torch.tensor(rel[graph_split_ids], dtype=torch.long).contiguous()

    # Create bi-directional graph
    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index=edge_index)
    data.entity = torch.from_numpy(uniq_entity)
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, len(uniq_entity), num_rels)
    data.samples = torch.from_numpy(samples)
    data.labels = torch.from_numpy(labels)

    return data

def build_test_graph(num_nodes, num_rels, triplets):
    src, rel, dst = triplets.transpose()

    src = torch.from_numpy(src)
    rel = torch.from_numpy(rel)
    dst = torch.from_numpy(dst)

    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index=edge_index)
    data.entity = torch.from_numpy(np.arange(num_nodes))
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, num_nodes, num_rels)

    return data

def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

# return MRR (filtered), and Hits @ (1, 3, 10)
def calc_mrr(embedding, w, test_triplets, all_triplets, hits=[]):
    with torch.no_grad():
        
        num_entity = len(embedding)

        ranks_s = []
        ranks_o = []

        head_relation_triplets = all_triplets[:, :2]
        tail_relation_triplets = torch.stack((all_triplets[:, 2], all_triplets[:, 1])).transpose(0, 1)

        for test_triplet in tqdm(test_triplets):

            # Perturb object
            subject = test_triplet[0]
            relation = test_triplet[1]
            object_ = test_triplet[2]

            subject_relation = test_triplet[:2]
            delete_index = torch.sum(head_relation_triplets == subject_relation, dim = 1)
            delete_index = torch.nonzero(delete_index == 2).squeeze()

            delete_entity_index = all_triplets[delete_index, 2].view(-1).numpy()
            perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
            perturb_entity_index = torch.from_numpy(perturb_entity_index)
            perturb_entity_index = torch.cat((perturb_entity_index, object_.view(-1)))
            
            emb_ar = embedding[subject] * w[relation]
            emb_ar = emb_ar.view(-1, 1, 1)

            emb_c = embedding[perturb_entity_index]
            emb_c = emb_c.transpose(0, 1).unsqueeze(1)
            
            out_prod = torch.bmm(emb_ar, emb_c)
            score = torch.sum(out_prod, dim = 0)
            score = torch.sigmoid(score)
            
            target = torch.tensor(len(perturb_entity_index) - 1)
            ranks_s.append(sort_and_rank(score, target))

            # Perturb subject
            object_ = test_triplet[2]
            relation = test_triplet[1]
            subject = test_triplet[0]

            object_relation = torch.tensor([object_, relation])
            delete_index = torch.sum(tail_relation_triplets == object_relation, dim = 1)
            delete_index = torch.nonzero(delete_index == 2).squeeze()

            delete_entity_index = all_triplets[delete_index, 0].view(-1).numpy()
            perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
            perturb_entity_index = torch.from_numpy(perturb_entity_index)
            perturb_entity_index = torch.cat((perturb_entity_index, subject.view(-1)))

            emb_ar = embedding[object_] * w[relation]
            emb_ar = emb_ar.view(-1, 1, 1)

            emb_c = embedding[perturb_entity_index]
            emb_c = emb_c.transpose(0, 1).unsqueeze(1)

            out_prod = torch.bmm(emb_ar, emb_c)
            score = torch.sum(out_prod, dim = 0)
            score = torch.sigmoid(score)

            target = torch.tensor(len(perturb_entity_index) - 1)
            ranks_o.append(sort_and_rank(score, target))

        ranks_s = torch.cat(ranks_s)
        ranks_o = torch.cat(ranks_o)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed

        ranks_s += 1
        ranks_o += 1

        mrr_s = torch.mean(1.0 / ranks_s.float())
        print("MRR Left(filtered): {:.6f}".format(mrr_s.item()))

        mrr_o = torch.mean(1.0 / ranks_o.float())
        print("MRR Right(filtered): {:.6f}".format(mrr_o.item()))

        mrr = torch.mean(1.0 / ranks.float())
        print("MRR (filtered): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            avg_count_s = torch.mean((ranks_s <= hit).float())
            avg_count_o = torch.mean((ranks_o <= hit).float())
            print("Hits Left(filtered) @ {}: {:.6f}".format(hit, avg_count_s.item()))
            print("Hits Right(filtered) @ {}: {:.6f}".format(hit, avg_count_o.item()))
            print("Hits (filtered) @ {}: {:.6f}".format(hit, avg_count.item()))
            
    return mrr.item()

def comp_deg_norm_dgl(g):
    g = g.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm

def node_norm_to_edge_norm_dgl(g, node_norm):
    g = g.local_var()
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
    return g.edata['norm']

def get_adj_and_degrees_dgl(num_nodes, triplets):
    """ Get adjacency list and degrees of the graph
    """
    adj_list = [[] for _ in range(num_nodes)]
    for i, triplet in enumerate(triplets):
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]
    return adj_list, degrees

def sample_edge_neighborhood_dgl(adj_list, degrees, n_triplets, sample_size):
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

def sample_edge_uniform_dgl(adj_list, degrees, n_triplets, sample_size):
    """Sample edges uniformly from all the edges."""
    all_edges = np.arange(n_triplets)
    return np.random.choice(all_edges, sample_size, replace=False)

def build_graph_from_triplets_dgl(num_nodes, num_rels, triplets):
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
    norm = comp_deg_norm_dgl(g)
    return g, rel.astype('int64'), norm.astype('int64')

def build_test_graph_dgl(num_nodes, num_rels, edges):
    src, rel, dst = edges.transpose()
    print("Test graph:")
    return build_graph_from_triplets_dgl(num_nodes, num_rels, (src, rel, dst))

def generate_sampled_graph_and_labels_dgl(triplets, sample_size, split_size,
                                      num_rels, adj_list, degrees,
                                      negative_rate, sampler="uniform"):
    """Get training graph and signals
    First perform edge neighborhood sampling on graph, then perform negative
    sampling to generate negative samples
    """
    # perform edge neighbor sampling
    if sampler == "uniform":
        edges = sample_edge_uniform_dgl(adj_list, degrees, len(triplets), sample_size)
    elif sampler == "neighbor":
        edges = sample_edge_neighborhood_dgl(adj_list, degrees, len(triplets), sample_size)
    else:
        raise ValueError("Sampler type must be either 'uniform' or 'neighbor'.")

    # relabel nodes to have consecutive node ids
    edges = triplets[edges]
    src, rel, dst = edges.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # negative sampling
    samples, labels = negative_sampling_dgl(relabeled_edges, len(uniq_v),
                                        negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)
    src = src[graph_split_ids]
    dst = dst[graph_split_ids]
    rel = rel[graph_split_ids]

    # build DGL graph
    g, rel, norm = build_graph_from_triplets_dgl(len(uniq_v), num_rels,
                                             (src, rel, dst))
    return g, uniq_v, rel, norm, samples, labels

def negative_sampling_dgl(pos_samples, num_entity, negative_rate):
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

def negative_sampling_v1(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.choice(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels

def generate_sampled_graph_and_labels_v1(triplets, sample_size, split_size, num_entity, num_rels, negative_rate):
    """
        Get training graph and signals
        First perform edge neighborhood sampling on graph, then perform negative
        sampling to generate negative samples
    """

    edges = sample_edge_uniform(len(triplets), sample_size)

    # Select sampled edges
    edges = triplets[edges]
    src, rel, dst = edges.transpose()
    uniq_entity, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # Negative sampling
    samples, labels = negative_sampling_v1(relabeled_edges, len(uniq_entity), negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)

    src = torch.tensor(src[graph_split_ids], dtype = torch.long).contiguous()
    dst = torch.tensor(dst[graph_split_ids], dtype = torch.long).contiguous()
    rel = torch.tensor(rel[graph_split_ids], dtype = torch.long).contiguous()

    # Create bi-directional graph
    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index = edge_index)
    data.entity = torch.from_numpy(uniq_entity)
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, len(uniq_entity), num_rels)
    data.samples = torch.from_numpy(samples)
    data.labels = torch.from_numpy(labels)

    return data

#######################################################################
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
        ranks.append(rank)
    return torch.LongTensor(ranks)

def calc_filtered_mrr(embedding, w, all_triplets, test_triplets, hits=[]):
    with torch.no_grad():
        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        o = test_triplets[:, 2]
        test_size = test_triplets.shape[0]

        triplets_to_filter = all_triplets.tolist()
        triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter}
        print('Perturbing subject...')
        ranks_s = perturb_s_and_get_filtered_rank(embedding, w, s, r, o, test_size, triplets_to_filter)
        print('Perturbing object...')
        ranks_o = perturb_o_and_get_filtered_rank(embedding, w, s, r, o, test_size, triplets_to_filter)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1  # change to 1-indexed

        ranks_s += 1
        ranks_o += 1

        mrr_s = torch.mean(1.0 / ranks_s.float())
        print("MRR Left(filtered): {:.6f}".format(mrr_s.item()))

        mrr_o = torch.mean(1.0 / ranks_o.float())
        print("MRR Right(filtered): {:.6f}".format(mrr_o.item()))

        mrr = torch.mean(1.0 / ranks.float())
        print("MRR (filtered): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            avg_count_s = torch.mean((ranks_s <= hit).float())
            avg_count_o = torch.mean((ranks_o <= hit).float())
            print("Hits Left(filtered) @ {}: {:.6f}".format(hit, avg_count_s.item()))
            print("Hits Right(filtered) @ {}: {:.6f}".format(hit, avg_count_o.item()))
            print("Hits (filtered) @ {}: {:.6f}".format(hit, avg_count.item()))

    return mrr.item()