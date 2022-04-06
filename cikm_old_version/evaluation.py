import torch
import numpy as np
import datetime
from logger_init import get_logger
from data_utils import batch_by_size
import time
#timer = CUDATimer()
logger = get_logger('eval', True, True, 'evaluation.txt')

# ranking_and_hits(model, Config.batch_size, valid_data, eval_h, eval_t,'dev_evaluation')
def ranking_and_hits(model, batch_size, dateset, eval_h, eval_t, name, X, adj_matrix):
    heads, rels, tails = dateset
    logger.info('')
    logger.info('-'*50)
    logger.info(name)
    logger.info('-'*50)
    logger.info('')
    hits_left = []
    hits_right = []
    hits = []
    ranks = []
    ranks_left = []
    ranks_right = []

    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    for bh, br, bt in batch_by_size(batch_size, heads, rels, tails):
        b_size = bh.size(0)
        bh = bh.cuda()
        br = br.cuda()
        bt = bt.cuda()
        pred1 = model.compute_rank(bh, br, X, adj_matrix)
        pred2 = model.compute_rank(bt, br, X, adj_matrix)

        e2_multi1 = torch.empty(b_size, pred1.size(1))
        e2_multi2 = torch.empty(b_size, pred1.size(1))

        for i, (h, r, t) in enumerate(zip(bh, br, bt)):
            e2_multi1[i] = eval_t[h.item(), r.item()].to_dense()
            e2_multi2[i] = eval_h[t.item(), r.item()].to_dense()
        e2_multi1 = e2_multi1.cuda()
        e2_multi2 = e2_multi2.cuda()

        for i in range(b_size):
            # save the prediction that is relevant
            target_value1 = pred1[i, bt[i].item()].item()
            target_value2 = pred2[i, bh[i].item()].item()
            # zero all known cases (this are not interesting)
            # this corresponds to the filtered setting
            pred1[i] += e2_multi1[i] * (-1e20)
            pred2[i] += e2_multi2[i] * (-1e20)
            # write base the saved values
            pred1[i][bt[i].item()] = target_value1
            pred2[i][bh[i].item()] = target_value2

        # sort and rank
        max_values, argsort1 = torch.sort(pred1, 1, descending=True)
        max_values, argsort2 = torch.sort(pred2, 1, descending=True)
        # max_values, argsort2 = torch.sort(pred2, 1, descending=True)
        for i in range(b_size):
            # find the rank of the target entities
            find_target1 = argsort1[i] == bt[i]
            find_target2 = argsort2[i] == bh[i]
            rank1 = torch.nonzero(find_target1)[0, 0].item() + 1
            rank2 = torch.nonzero(find_target2)[0, 0].item() + 1
            # rank+1, since the lowest rank is rank 1 not rank 0
            # ranks.append(rank1+1)
            ranks_left.append(rank1)
            # ranks.append(rank2+1)
            ranks_right.append(rank2)

            # this could be done more elegantly, but here you go
            hits[0].append(int(rank1 <= 1))
            hits[0].append(int(rank2 <= 1))
            hits_left[0].append((int(rank1 <= 1)))
            hits_right[0].append((int(rank2 <= 1)))

            hits[2].append(int(rank1 <= 3))
            hits[2].append(int(rank2 <= 3))
            hits_left[2].append((int(rank1 <= 3)))
            hits_right[2].append((int(rank2 <= 3)))

            hits[9].append(int(rank1 <= 10))
            hits[9].append(int(rank2 <= 10))
            hits_left[9].append((int(rank1 <= 10)))
            hits_right[9].append((int(rank2 <= 10)))
            # for hits_level in range(10):
            #     if rank1 <= hits_level:
            #         hits[hits_level].append(1.0)
            #         hits_left[hits_level].append(1.0)
            #     else:
            #         hits[hits_level].append(0.0)
            #         hits_left[hits_level].append(0.0)
            #
            #     if rank2 <= hits_level:
            #         hits[hits_level].append(1.0)
            #         hits_right[hits_level].append(1.0)
            #     else:
            #         hits[hits_level].append(0.0)
            #         hits_right[hits_level].append(0.0)

    # for i in range(10):
    #     logger.info('Hits left @{0}: {1}'.format(i+1, np.mean(hits_left[i])))
    #     logger.info('Hits right @{0}: {1}'.format(i+1, np.mean(hits_right[i])))

    logger.info('Hits @{0}: {1}'.format(1, np.mean(hits_left[0])))
    # logger.info('Hits left @{0}: {1}'.format(1, np.mean(hits_left[0])))
    # logger.info('Hits right @{0}: {1}'.format(1, np.mean(hits_right[0])))
    # logger.info('Hits @{0}: {1}'.format(1, np.mean(hits[0])))

    logger.info('Hits @{0}: {1}'.format(3, np.mean(hits_left[2])))
    # logger.info('Hits left @{0}: {1}'.format(3, np.mean(hits_left[2])))
    # logger.info('Hits right @{0}: {1}'.format(3, np.mean(hits_right[2])))
    # logger.info('Hits @{0}: {1}'.format(3, np.mean(hits[2])))

    logger.info('Hits @{0}: {1}'.format(10, np.mean(hits_left[9])))
    # logger.info('Hits left @{0}: {1}'.format(10, np.mean(hits_left[9])))
    # logger.info('Hits right @{0}: {1}'.format(10, np.mean(hits_right[9])))
    # logger.info('Hits @{0}: {1}'.format(10, np.mean(hits[9])))

    logger.info('Mean rank: {0}'.format(np.mean(ranks_left)))
    # logger.info('Mean rank left: {0}'.format(np.mean(ranks_left)))
    # logger.info('Mean rank right: {0}'.format(np.mean(ranks_right)))
    # logger.info('Mean rank: {0}'.format(np.mean(ranks_left+ranks_right)))
    logger.info('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks_left))))
    # logger.info('Mean reciprocal rank left: {0}'.format(np.mean(1./np.array(ranks_left))))
    # logger.info('Mean reciprocal rank right: {0}'.format(np.mean(1./np.array(ranks_right))))
    # logger.info('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks_left+ranks_right))))

'''direct为true时为正例集，direct为false时为负例集'''
def triple_classification(model, batch_size, dateset, eval_h, eval_t, name, threshold, direct, X, adj_matrix):
    heads, rels, tails = dateset
    logger.info('')
    logger.info('-'*50)
    logger.info(name)
    logger.info('-'*50)
    logger.info('')
    correct_num = 0.0
    error_num = 0.0
    for bh, br, bt in batch_by_size(batch_size, heads, rels, tails):
        b_size = bh.size(0)
        bh = bh.cuda()
        br = br.cuda()
        bt = bt.cuda()
        pred1 = model.compute_rank(bh, br, X, adj_matrix)
        pred2 = model.compute_rank(bt, br, X, adj_matrix)

        e2_multi1 = torch.empty(b_size, pred1.size(1))
        e2_multi2 = torch.empty(b_size, pred1.size(1))

        for i, (h, r, t) in enumerate(zip(bh, br, bt)):
            e2_multi1[i] = eval_t[h.item(), r.item()].to_dense()
            e2_multi2[i] = eval_h[t.item(), r.item()].to_dense()

        for i in range(b_size):
            target_value1 = pred1[i, bt[i].item()].item()
            target_value2 = pred2[i, bh[i].item()].item()
            if direct is True:
                if target_value1 > threshold or target_value2 >threshold:
                    correct_num = correct_num + 1.0
                else:
                    error_num = error_num + 1.0
            else:
                if target_value1 < threshold or target_value2 < threshold:
                    correct_num = correct_num + 1.0
                else:
                    error_num = error_num + 1.0
    precision = correct_num / (correct_num + error_num)
    return precision, correct_num, error_num
