import torch
import numpy as np
import argparse
import os
import time
from datetime import datetime
from config import config
from logger_init import get_logger
from read_data import index_ent_rel, graph_size, read_data, read_data_with_rel_reverse, read_virtual_data, read_virtual_data_dict, read_virtual_data_with_pre
from data_utils import inplace_shuffle, heads_tails, batch_by_num, batch_by_size, get_adjacent_matrix, train_step
from evaluation import ranking_and_hits, triple_classification
from model import ConvE, DistMult, Complex, VN_WGCN, GraphConvolution, VN_WGCN_TransE

np.set_printoptions(precision=3)
logger = get_logger('train', True)
logger.info('START TIME : {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
Config = config()
#model_name = 'DistMult_{0}_{1}'.format(Config.input_dropout, Config.dropout)
model_name = '{2}_{0}_{1}'.format(Config.input_dropout, Config.dropout, Config.model_name)
load = False
if Config.dataset is None:
    Config.dataset = 'FB15k-237'
save_dir = 'saved_models'
if not os.path.exists(save_dir): os.makedirs(save_dir)
model_path = 'saved_models/{0}_{1}.model'.format(Config.dataset, model_name)

task_dir = config().task_dir
'''此处为读数据，kd_index是字典'''
kb_index = index_ent_rel(os.path.join(task_dir, 'train.txt'),
                         os.path.join(task_dir, 'valid.txt'),
                         os.path.join(task_dir, '_aux.txt'),
                         os.path.join(task_dir, 'test.txt'))
'''获得边和实体个数'''
n_ent, n_rel = graph_size(kb_index)
'''获得头结点列表、关系列表、尾结点列表，列表内均为id'''
# train_data_with_reverse = read_data_with_rel_reverse(os.path.join(task_dir, 'train.txt'), kb_index)
# inplace_shuffle(*train_data_with_reverse)
# aux_data_with_reverse = read_data_with_rel_reverse(os.path.join(task_dir, '_aux.txt'), kb_index)
# inplace_shuffle(*aux_data_with_reverse)
# heads, tails = heads_tails(n_ent, train_data_with_reverse, aux_data_with_reverse)
'''获得头结点列表、关系列表、尾结点列表，列表内均为id'''
train_data = read_data(os.path.join(task_dir, 'train.txt'), kb_index)
valid_data = read_data(os.path.join(task_dir, 'valid.txt'), kb_index)
test_data = read_data(os.path.join(task_dir, 'test.txt'), kb_index)
aux_data = read_data(os.path.join(task_dir, '_aux.txt'), kb_index)

heads, tails = heads_tails(n_ent, train_data)
aux_heads, aux_tails = heads_tails(n_ent, train_data, aux_data)

inplace_shuffle(*train_data)
inplace_shuffle(*aux_data)
'''获得邻接矩阵'''
adj_matrix = get_adjacent_matrix(n_ent, n_rel, train_data, aux_data)
X = torch.LongTensor([i for i in range(n_ent)]).cuda()

'''此处再加一个virtual_neighbor'''
if Config.enable_virtual is True:
    virtual_data = read_virtual_data(os.path.join(task_dir, 'virtual.txt'), kb_index)
    lamda_dict_head, lamda_dict_rel, lamda_dict_tail, rule_ground_dict, lamda_dict = read_virtual_data_dict(
        os.path.join(task_dir, 'virtual.txt'), kb_index)
    virtual_data_with_pre = read_virtual_data_with_pre(os.path.join(task_dir, 'virtual.txt'), kb_index, lamda_dict)
    eval_h, eval_t = heads_tails(n_ent, train_data, valid_data, test_data, virtual_data)
    virtual_data = [torch.LongTensor(vec) for vec in virtual_data]
    virtual_data_with_pre = [torch.LongTensor(vec) for vec in virtual_data_with_pre]
else:
    eval_h, eval_t = heads_tails(n_ent, train_data, valid_data, test_data)

'''三个张量，分别为头实体、关系、尾实体id号组成的张量列表'''
valid_data = [torch.LongTensor(vec) for vec in valid_data]
test_data = [torch.LongTensor(vec) for vec in test_data]
train_data = [torch.LongTensor(vec) for vec in train_data]
aux_data = [torch.LongTensor(vec) for vec in aux_data]
# train_data_with_reverse = [torch.LongTensor(vec) for vec in train_data_with_reverse]
# aux_data_with_reverse = [torch.LongTensor(vec) for vec in aux_data_with_reverse]

'''此处获得virtual_neighbor的共同之处， 包括每个batch的数据以及共同规则集之和'''


parser = argparse.ArgumentParser(description='VN_WGCN argparse')
parser.add_argument('--model_name', type=str, default='VN_WGCN', help='specific the model name')

args = parser.parse_args()

def main():
    if args.model_name == 'VN_WGCN':
        model = VN_WGCN(n_ent, n_rel)
    elif args.model_name == 'ConvE':
        model = ConvE(n_ent, n_rel)
    elif args.model_name == 'DistMult':
        model = DistMult(n_ent, n_rel)
    elif args.model_name == 'ComplEx':
        model = Complex(n_ent, n_rel)
    else:
        logger.info('Unknown model: {0}', args.model_name)
        raise Exception("Unknown model!")

    if Config.cuda:
        model.cuda()
    model.init()
    params = [value.numel() for value in model.parameters()]
    print(params)
    print(sum(params))
    opt = torch.optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.L2)
    for epoch in range(Config.pre_epochs):
        model.train()
        train_step(train_data, opt, model, X, adj_matrix, 'train', n_ent)
    for epoch in range(Config.epochs):
        logger.info('epoch {0}'.format(epoch))
        start = time.time()
        train_step(aux_data, opt, model, X, adj_matrix, 'aux', n_ent)
        if Config.hard_rule is True:
            train_step(virtual_data, opt, model, X, adj_matrix, 'virtual', n_ent)
        if Config.hard_rule is False:
            tnorm = torch.zeros(1, 400).cuda()
            for key in lamda_dict_head:
                lamda_head = torch.LongTensor(lamda_dict_head[key]).cuda()
                lamda_rel = torch.LongTensor(lamda_dict_rel[key]).cuda()
                lamda_tail = torch.LongTensor(lamda_dict_tail[key]).cuda()
                sum_adj = model.get_triple_score(lamda_head, lamda_rel, lamda_tail)
                sum_adj = torch.sum(sum_adj).cuda() * Config.C * float(key)
                tnorm[0][lamda_dict[key]] = sum_adj
            tnorm = tnorm.squeeze()
            vh, vr, vt, vph, vpr, vpt, lamda_id = virtual_data_with_pre
            n_virtual = vh.size(0)
            rand_idx = torch.randperm(n_virtual)
            vh = vh[rand_idx].cuda()
            vr = vr[rand_idx].cuda()
            vt = vt[rand_idx].cuda()
            vph = vph[rand_idx].cuda()
            vpr = vpr[rand_idx].cuda()
            vpt = vpt[rand_idx].cuda()
            lamda_id = lamda_id[rand_idx].cuda()
            tot = 0.0
            for bh, br, bt, vbh, vbr, vbt, lid in batch_by_num(Config.n_batch, vh, vr, vt, vph, vpr, vpt, lamda_id):
                opt.zero_grad()
                batch_size = bh.size(0)
                bh = bh.repeat(1, Config.negative_sample_size + 1).squeeze()
                br = br.repeat(1, Config.negative_sample_size + 1).squeeze()
                negative_bt = torch.LongTensor(1, Config.negative_sample_size * batch_size).random_(n_ent - 1).squeeze().cuda()
                bt = torch.cat((bt, negative_bt), 0)
                pred = model.forward(bh, br, bt, X, adj_matrix)
                e2_multi_negative = torch.zeros(1, Config.negative_sample_size * batch_size).cuda()
                pred1 = model.get_triple_score(vbh, vbr, vbt)
                batch_norm = tnorm[lid]
                e2_multi = pred1 + batch_norm
                e2_multi_norm = torch.ones(e2_multi.size(0))
                e2_multi = torch.min(e2_multi_norm.cuda(), e2_multi.cuda()).cuda()
                e2_multi = torch.cat((e2_multi, e2_multi_negative.squeeze()), 0).cuda()
                e2_multi = e2_multi.detach()
                loss = model.loss(pred, e2_multi)
                loss.backward()
                opt.step()
                batch_loss = torch.sum(loss)
                tot += bh.size(0)
                print('\r{:>10} which {} progress {} loss: {}'.format('', 'virtual', tot / n_virtual, batch_loss), end='')
            print("\r virtual progress finished")
        logger.info('')
        end = time.time()
        time_used = end - start
        logger.info('one epoch time: {} minutes'.format(time_used/60))
        logger.info('saving to {0}'.format(model_path))
        torch.save(model.state_dict(), model_path)

        model.eval()
        with torch.no_grad():
            start = time.time()
            # ranking_and_hits(model, Config.batch_size, valid_data, eval_h, eval_t, 'dev_evaluation', X, adj_matrix)
            end = time.time()
            logger.info('eval time used: {} minutes'.format((end - start)/60))
            ranking_and_hits(model, Config.batch_size, test_data, eval_h, eval_t, 'test_evaluation', X, adj_matrix)
if __name__ == '__main__':
    main()
