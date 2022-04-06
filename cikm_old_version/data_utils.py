from random import randint
from collections import defaultdict
import torch
from config import config
Config = config()

def heads_tails(n_ent, train_data, valid_data=None, test_data=None, virtual_data=None):
    train_src, train_rel, train_dst = train_data
    if valid_data:
        valid_src, valid_rel, valid_dst = valid_data
    else:
        valid_src = valid_rel = valid_dst = []
    if test_data:
        test_src, test_rel, test_dst = test_data
    else:
        test_src = test_rel = test_dst = []
    if virtual_data:
        virtual_src, virtual_rel, virtual_dst = virtual_data
    else:
        virtual_src = virtual_rel = virtual_dst = []
    all_src = train_src + valid_src + test_src + virtual_src
    all_rel = train_rel + valid_rel + test_rel + virtual_rel
    all_dst = train_dst + valid_dst + test_dst + virtual_dst
    heads = defaultdict(lambda: set())
    tails = defaultdict(lambda: set())
    for s, r, t in zip(all_src, all_rel, all_dst):
        tails[s, r].add(t)
        heads[t, r].add(s)
    heads_sp = {}
    tails_sp = {}
    for k in tails.keys():
        '''>>> i = torch.LongTensor([[0, 1, 1],
                          [2, 0, 2]])
           >>> v = torch.FloatTensor([3, 4, 5])
           >>> torch.sparse.FloatTensor(i, v, torch.Size([2,3])).to_dense()
            0  0  3
            4  0  5
        '''
        tails_sp[k] = torch.sparse.FloatTensor(torch.LongTensor([list(tails[k])]),
                                               torch.ones(len(tails[k])), torch.Size([n_ent]))
        # print(tails_sp[k])
    for k in heads.keys():
        heads_sp[k] = torch.sparse.FloatTensor(torch.LongTensor([list(heads[k])]),
                                               torch.ones(len(heads[k])), torch.Size([n_ent]))
    return heads_sp, tails_sp

def inplace_shuffle(*lists):
    idx = []
    '''lists[0]是三元组源点列表'''
    for i in range(len(lists[0])):
        idx.append(randint(0, i))
    for ls in lists:
        '''item是标号'''
        for i, item in enumerate(ls):
            j = idx[i]
            ls[i], ls[j] = ls[j], ls[i]

def batch_by_num(n_batch, *lists, n_sample=None):
    if n_sample is None:
        n_sample = len(lists[0])
    for i in range(n_batch):
        head = int(n_sample * i / n_batch)
        tail = int(n_sample * (i + 1) / n_batch)
        ret = [ls[head:tail] for ls in lists]
        if len(ret) > 1:
            yield ret
        else:
            yield ret[0]

def batch_by_size(batch_size, *lists, n_sample=None):
    if n_sample is None:
        n_sample = len(lists[0])
    head = 0
    while head < n_sample:
        tail = min(n_sample, head + batch_size)
        ret = [ls[head:tail] for ls in lists]
        head += batch_size
        if len(ret) > 1:
            yield ret
        else:
            yield ret[0]

def get_adjacent_matrix(n_ent, n_rel, train_data_with_reverse, aux_data_with_reverse=None):
    train_src, train_rel, train_dst = train_data_with_reverse
    if aux_data_with_reverse:
        aux_src, aux_rel, aux_dst = aux_data_with_reverse
    else:
        aux_src = aux_rel = aux_dst = []
    all_src = train_src + aux_src
    all_rel = train_rel + aux_rel
    all_dst = train_dst + aux_dst
    all_src = all_src + [i for i in range(n_ent)]
    all_dst = all_dst + [i for i in range(n_ent)]
    all_rel = all_rel + [n_rel for i in range(n_ent)]
    indices = torch.LongTensor([all_src, all_dst]).cuda()
    v = torch.LongTensor(all_rel).cuda()
    adjacencies = [indices, v, n_ent]
    return adjacencies

def train_step(dataset, opt, model, X, adj_matrix, flag, n_ent):
    h, r, t = dataset
    n_train = h.size(0)
    rand_idx = torch.randperm(n_train)
    h = h[rand_idx].cuda()
    r = r[rand_idx].cuda()
    t = t[rand_idx].cuda()
    tot = 0.0
    for bh, br, bt in batch_by_num(Config.n_batch, h, r, t):
        opt.zero_grad()
        batch_size = bh.size(0)
        e2_multi_positive = torch.ones(1, batch_size).cuda()
        e2_multi_negative = torch.zeros(1, Config.negative_sample_size * batch_size).cuda()
        e2_multi = torch.cat((e2_multi_positive, e2_multi_negative), 1).cuda()
        bh = bh.repeat(1, Config.negative_sample_size + 1).squeeze()
        br = br.repeat(1, Config.negative_sample_size + 1).squeeze()
        negative_bt = torch.LongTensor(1, Config.negative_sample_size * batch_size).random_(n_ent - 1).squeeze().cuda()
        bt = torch.cat((bt, negative_bt), 0)
        pred = model.forward(bh, br, bt, X, adj_matrix)
        loss = model.loss(pred, e2_multi.squeeze())
        loss.backward()
        opt.step()
        batch_loss = torch.sum(loss)
        tot += bh.size(0)
        print('\r{:>10} which {} progress {} loss: {}'.format('', flag, tot / n_train, batch_loss), end='')