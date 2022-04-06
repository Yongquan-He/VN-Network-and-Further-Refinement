import argparse
import torch.nn as nn
import torch.nn.functional as F
from dgl.data.knowledge_graph import load_data
from torch.nn import Parameter

from models.model_main import RelGraphConv2
from models.model_main import BaseRGCN
from models.rule_confidence_main import RuleConfidence, get_soft_labels_for_one_hop, get_soft_labels_for_two_hop
from utils.utils_use import *


class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, h_dim)

    def forward(self, g, h, r, norm):
        return self.embedding(h.squeeze())


class RGCN(BaseRGCN):
    def build_input_layer(self):
        return EmbeddingLayer(self.num_nodes, self.h_dim)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return RelGraphConv2(self.h_dim, self.h_dim, self.num_rels, "bdd",
                            self.num_bases, activation=act, self_loop=True,
                            dropout=self.dropout)

class DistMult(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0, isSigmoid=False):
        super(DistMult, self).__init__()
        self.rgcn = RGCN(in_dim, h_dim, h_dim, num_rels * 2, num_bases,
                         num_hidden_layers, dropout, use_cuda)
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))
        self.isSigmoid = isSigmoid

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:, 0]]
        r = self.w_relation[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        score = torch.sum(s * r * o, dim=1)
        if self.isSigmoid:
            score = F.sigmoid(score)
        return score

    def forward(self, g, h, r, norm):
        return self.rgcn.forward(g, h, r, norm)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, triplets)
        if self.isSigmoid:
            predict_loss = F.binary_cross_entropy(score, labels)
        else:
            predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss

class TransE(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0, isSigmoid=False):
        super(TransE, self).__init__()
        self.rgcn = RGCN(in_dim, h_dim, h_dim, num_rels * 2, num_bases,
                         num_hidden_layers, dropout, use_cuda)
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))
        self.isSigmoid = isSigmoid

    def calc_score(self, embedding, triplets):
        s = embedding[triplets[:, 0]]
        r = self.w_relation[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        score = torch.sum(s + r - o, dim=1)
        if self.isSigmoid:
            score = F.sigmoid(score)
        return score

    def forward(self, g, h, r, norm):
        return self.rgcn.forward(g, h, r, norm)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, triplets)
        if self.isSigmoid:
            predict_loss = F.binary_cross_entropy(score, labels)
        else:
            predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss

class ConvE(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0, isSigmoid=False):
        super(ConvE, self).__init__()
        self.rgcn = RGCN(in_dim, h_dim, h_dim, num_rels * 2, num_bases,
                         num_hidden_layers, dropout, use_cuda)
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))
        self.isSigmoid = isSigmoid

        self.hidden_drop = torch.nn.Dropout(dropout)
        self.feature_map_drop = torch.nn.Dropout2d(dropout)
        self.emb_dim1 = 20
        self.emb_dim2 = h_dim // self.emb_dim1

        self.conv = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=True)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(h_dim)
        self.register_parameter('b', Parameter(torch.zeros(in_dim)))
        self.fc = torch.nn.Linear(9728, h_dim)

    def calc_score(self, embedding, triplets):
        s = embedding[triplets[:, 0]]
        r = self.w_relation[triplets[:, 1]]
        o = embedding[triplets[:, 2]]

        s = s.view(-1, 1, self.emb_dim1, self.emb_dim2)
        r = r.view(-1, 1, self.emb_dim1, self.emb_dim2)

        stacked_inputs = torch.cat([s, r], 2)

        x = self.bn0(stacked_inputs)
        x = self.conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        score = x * o
        if self.isSigmoid:
            score = F.sigmoid(score)
        return score

    def forward(self, g, h, r, norm):
        return self.rgcn.forward(g, h, r, norm)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, triplets)
        if self.isSigmoid:
            predict_loss = F.binary_cross_entropy(score, labels)
        else:
            predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss

class Analogy(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0, isSigmoid=False):
        super(Analogy, self).__init__()
        self.rgcn = RGCN(in_dim, h_dim, h_dim, num_rels * 2, num_bases,
                         num_hidden_layers, dropout, use_cuda)
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))
        self.isSigmoid = isSigmoid
        self.dim = h_dim

    def calc_score(self, embedding, triplets):
        # triples: [None, 3, dim]
        # h,r,t: [None, 1, dim] -> [None, dim]
        # [100 + 50(x) + 50(y)]
        # x, y
        # -y x
        h = embedding[triplets[:, 0]]
        r = self.w_relation[triplets[:, 1]]
        t = embedding[triplets[:, 2]]
        # h_scalar: [None, dim/2]
        # h_x, h_y: [None, dim/4]
        h_scalar, h_x, h_y = self.split_embedding(h)
        r_scalar, r_x, r_y = self.split_embedding(r)
        t_scalar, t_x, t_y = self.split_embedding(t)
        # score_scalar: [None]
        score_scalar = torch.sum(h_scalar * r_scalar * t_scalar, dim=1)
        # score_block: [None]
        score_block = torch.sum(h_x * r_x * t_x
                                    + h_x * r_y * t_y
                                    + h_y * r_x * t_y
                                    - h_y * r_y * t_x, dim=1)
        # score: [None]
        score = score_scalar + score_block
        if self.isSigmoid:
            score = F.sigmoid(score)
        return score

    def split_embedding(self, embedding):
        # embedding: [None, dim]
        assert self.dim % 4 == 0
        num_scalar = self.dim // 2
        num_block = self.dim // 4
        embedding_scalar = embedding[:, 0:num_scalar]
        embedding_x = embedding[:, num_scalar:-num_block]
        embedding_y = embedding[:, -num_block:]
        return embedding_scalar, embedding_x, embedding_y

    def forward(self, g, h, r, norm):
        return self.rgcn.forward(g, h, r, norm)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, triplets)
        if self.isSigmoid:
            predict_loss = F.binary_cross_entropy(score, labels)
        else:
            predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss

def node_norm_to_edge_norm(g, node_norm):
    g = g.local_var()
    # convert to edge norm
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
    return g.edata['norm']


def main(args):

    target_dir = os.path.join('./data', args.data, args.sub_data)

    ####################################################################
    # logger to log
    logger_path = os.path.join(target_dir, 'log/')
    makedirs(logger_path)
    logger = get_logger(logger_path, "rgcn-%s.log" % timestamp())
    ####################################################################

    # set gpu
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not use_cuda:
            print('WARNING: You have a CUDA device, so you should probably run with --cuda')
        else:
            torch.cuda.manual_seed_all(args.seed)
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    # load data and set model
    model_name = '{0}_{1}'.format(args.sub_data, args.model)
    makedirs(os.path.join(target_dir, 'saved_models/'))
    model_path = os.path.join(target_dir, 'saved_models/{0}_{1}.model'.format(args.data, model_name))

    entity2id, relation2id, train_data, valid_data, aux_data, test_data = load_data(target_dir)
    num_nodes = len(entity2id)
    num_rels = len(relation2id)

    triples_set = set()
    triples_set.update(train_data)
    triples_set.update(valid_data)
    triples_set.update(aux_data)
    another_set = set(test_data)

    one_hop_pre, one_hop_ground, two_hop_pre1, two_hop_pre2, two_hop_ground = load_vn_data(target_dir, triples_set, another_set)
    two_hop_pre1.append((1, 1, 1))
    two_hop_pre2.append((1, 1, 1))
    two_hop_ground.append((1, 1, 1))
    train_data = np.array(train_data)
    aux_data = np.array(aux_data)
    valid_data = np.array(valid_data)
    test_data = np.array(test_data)
    one_hop_pre = np.array(one_hop_pre)
    one_hop_ground = np.array(one_hop_ground)
    two_hop_pre1 = np.array(two_hop_pre1)
    two_hop_pre2 = np.array(two_hop_pre2)
    two_hop_ground = np.array(two_hop_ground)

    one_hop_ground_rule = []
    two_hop_ground_rule = []
    for ground, pre in zip(one_hop_ground, one_hop_pre):
        one_hop_ground_rule.append(check_rule(ground, pre))
    for ground, pre1, pre2 in zip(two_hop_ground, two_hop_pre1, two_hop_pre2):
        two_hop_ground_rule.append(check_rule(ground, pre1, pre2))
    one_hop_ground_rule_set = set(one_hop_ground_rule)
    two_hop_ground_rule_set = set(two_hop_ground_rule)

    all_data = torch.LongTensor(np.concatenate((train_data, valid_data, aux_data, test_data, one_hop_ground, two_hop_ground)))

    test_data = torch.LongTensor(test_data)

    # construct the total graph
    total_data = np.concatenate([train_data, aux_data, one_hop_ground, two_hop_ground])
    total_graph, total_rel, total_norm = build_test_graph(num_nodes, num_rels, total_data)
    total_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    total_rel = torch.from_numpy(total_rel)
    total_norm = node_norm_to_edge_norm(total_graph, torch.from_numpy(total_norm).view(-1, 1))

    premise_data = np.concatenate([one_hop_pre, two_hop_pre1, two_hop_pre2])
    virtual_data = np.concatenate([one_hop_ground, two_hop_ground])

    print("total length of virtual data: %d" % (virtual_data.shape[0]))

    # create model
    if args.model is None:
        model = DistMult(num_nodes,
                        args.embedding_dim,
                        num_rels,
                        num_bases=args.n_bases,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        use_cuda=use_cuda,
                        reg_param=args.regularization,
                        isSigmoid=args.isSigmoid)
    elif args.model == 'conve':
        model = ConvE(num_nodes,
                       args.embedding_dim,
                       num_rels,
                       num_bases=args.n_bases,
                       num_hidden_layers=args.n_layers,
                       dropout=args.dropout,
                       use_cuda=use_cuda,
                       reg_param=args.regularization,
                       isSigmoid=args.isSigmoid)
    elif args.model == 'distmult':
        model = DistMult(num_nodes,
                       args.embedding_dim,
                       num_rels,
                       num_bases=args.n_bases,
                       num_hidden_layers=args.n_layers,
                       dropout=args.dropout,
                       use_cuda=use_cuda,
                       reg_param=args.regularization,
                       isSigmoid=args.isSigmoid)
    elif args.model == 'analogy':
        model = Analogy(num_nodes,
                       args.embedding_dim,
                       num_rels,
                       num_bases=args.n_bases,
                       num_hidden_layers=args.n_layers,
                       dropout=args.dropout,
                       use_cuda=use_cuda,
                       reg_param=args.regularization,
                       isSigmoid=args.isSigmoid)
    elif args.model == 'transe':
        model = TransE(num_nodes,
                       args.embedding_dim,
                       num_rels,
                       num_bases=args.n_bases,
                       num_hidden_layers=args.n_layers,
                       dropout=args.dropout,
                       use_cuda=use_cuda,
                       reg_param=args.regularization,
                       isSigmoid=args.isSigmoid)
    else:
        print('Unknown model: {0}', args.model)
        raise Exception("Unknown model!")

    if use_cuda:
        model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    epoch = 0
    aux_epoch = 0
    best_mrr = 0

    # training loop
    print("start training...")

    ts = time.time()
    logger.info("\n\npretrain the aux triples...")
    if args.n_epochs_aux:
        print("----------------------------------------")
        print("Pretrain the aux data")
        while True:
            model.train()
            aux_epoch += 1
            #
            batch_train_step(ts, logger, aux_epoch, best_mrr, model, args, aux_data, num_rels, optimizer)
            if aux_epoch > args.n_epochs_aux:
                break
        torch.cuda.empty_cache()
    logger.info("Done pretrain aux triples. [%.1f s]" % (time.time() - ts))

    if args.n_epochs_aux != 1:
        logger.info("\n\nTest for aux pretrained model")
        ts = time.time()
        print("----------------------------------------")
        print("Test for the aux pretrained model")
        if use_cuda:
            model.cpu()
        model.eval()
        print("start eval")

        total_graph = total_graph.cpu()
        total_node_id = total_node_id.cpu()
        total_rel = total_rel.cpu()
        total_norm = total_norm.cpu()

        embed = model(total_graph, total_node_id, total_rel, total_norm)
        # validation
        mrr_s, mrr_o, mrr, hits_s, hits_o, hits = calc_mrr(embed, model.w_relation, torch.LongTensor(total_data),
                             torch.LongTensor(valid_data), torch.LongTensor(test_data), hits=[1, 3, 10], eval_bz=args.eval_batch_size,
                             eval_p=args.eval_protocol)
        best_mrr = mrr

        logger.info("Done test for aux pretrained model: MRR_S {:.4f} | MRR_S {:.4f} | MRR_S {:.4f} "
                    "| Hits_s(1, 3, 10) {} | Hits_o {} | Hits {} [{:.1f} s]".
                    format(mrr_s, mrr_o, mrr, hits_s, hits_o, hits, time.time() - ts))

    print("----------------------------------------")
    print("Start to train the aux and virtual data")
    logger.info("\n\n Start to train the aux and inferred triples...")
    ts = time.time()
    while True:
        model.train()
        epoch += 1
        model.cuda()

        if args.data == 'yago37':
            edges = sample_edge_uniform(train_data.shape[0], args.train_graph_size)
            train_graph = train_data[edges]
        else:
            train_graph = train_data
        vp_data = np.concatenate([virtual_data, aux_data, premise_data, train_graph])

        src, rel, dst = vp_data.transpose()
        uniq_v, edges = np.unique((src, dst), return_inverse=True)
        src, dst = np.reshape(edges, (2, -1))
        relabeled_total_edges = np.stack((src, rel, dst)).transpose()
        relabeled_virtual_data = relabeled_total_edges[:virtual_data.shape[0]]
        relabeled_one_hop_ground = relabeled_virtual_data[:one_hop_ground.shape[0]]
        relabeled_two_hop_ground = relabeled_virtual_data[-two_hop_ground.shape[0]:]
        relabeled_aux_data = relabeled_total_edges[virtual_data.shape[0]: virtual_data.shape[0] + aux_data.shape[0]]
        relabeled_one_hop_pre = relabeled_total_edges[virtual_data.shape[0] + aux_data.shape[0]: virtual_data.shape[0] + aux_data.shape[0] + one_hop_pre.shape[0]]
        relabeled_two_hop_pre1 = relabeled_total_edges[virtual_data.shape[0] + aux_data.shape[0] + one_hop_pre.shape[0]: virtual_data.shape[0] + aux_data.shape[0] + one_hop_pre.shape[0] + two_hop_pre1.shape[0]]
        relabeled_two_hop_pre2 = relabeled_total_edges[-two_hop_pre2.shape[0] - train_graph.shape[0]: -train_graph.shape[0]]

        logger.info("Done relabeled data [%.1f s]" % (time.time() - ts))

        print("# virtual and premise data nodes: {}".format(len(uniq_v)))
        print("# virtual and premise data edges: {}".format(len(src) * 2))

        # Build graph
        g, rel, norm = build_graph_from_triplets(len(uniq_v), num_rels, (src, rel, dst))
        # Set node/edge feature
        node_id = torch.from_numpy(uniq_v).view(-1, 1).long()
        edge_type = torch.from_numpy(rel)
        edge_norm = node_norm_to_edge_norm(g, torch.from_numpy(norm).view(-1, 1))
        deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)
        node_id, deg = node_id.cuda(), deg.cuda()
        g = g.to(args.gpu)
        edge_type, edge_norm = edge_type.cuda(), edge_norm.cuda()
        logger.info("Done build graph [%.1f s]" % (time.time() - ts))

        with torch.no_grad():
            rule_confidence = RuleConfidence(args, model.w_relation,
                                  one_hop_ground_rule, one_hop_ground_rule_set,
                                  two_hop_ground_rule, two_hop_ground_rule_set)
            one_hop_confidence = torch.stack(rule_confidence.get_confidence_for_one_hop_rule())
            two_hop_confidence = torch.stack(rule_confidence.get_confidence_for_two_hop_rule())


            embed = model(g, node_id, edge_type, edge_norm)
            if not args.hard_label:
                one_hop_labels = get_soft_labels_for_one_hop(relabeled_one_hop_pre, relabeled_one_hop_ground, one_hop_confidence, args.penalty,
                                                         model, embed, args)
                two_hop_labels = get_soft_labels_for_two_hop(relabeled_two_hop_pre1, relabeled_two_hop_pre2, relabeled_two_hop_ground, two_hop_confidence,
                                                         args.penalty, model, embed, args)
            else:
                one_hop_labels = torch.ones(len(relabeled_one_hop_pre)).cuda()
                two_hop_labels = torch.ones(len(relabeled_two_hop_pre1)).cuda()
            soft_labels = torch.cat([one_hop_labels, two_hop_labels]).detach()

            logger.info("Done soft labels [%.1f s]" % (time.time() - ts))
            logger.info("Some rule confidence: {}".format(one_hop_confidence[:min(5, one_hop_confidence.shape[0])].tolist()))
            logger.info("Some soft labels: {}".format(soft_labels[:min(5, soft_labels.shape[0])].tolist()))

        # train by batch
        batch = 0
        logger.info("\n\n Train by batch...")
        for batch_soft_labels, batch_virtual_data, batch_aux_data in batch_by_size_softlabels(args.batch_size, relabeled_virtual_data, relabeled_aux_data, soft_labels):
            batch += 1
            t0 = time.time()
            embed = model(g, node_id, edge_type, edge_norm)

            gpu_usage = get_gpu_memory_map()[int(args.gpu)]
            # logger.info('GPU usage for embeddings %d' % gpu_usage)

            td = time.time()
            print("Time cost: %4f" % (td - ts))

            # sampling
            samples, neg_labels = negative_sampling_for_virtual_data(np.concatenate([batch_virtual_data, batch_aux_data]), len(uniq_v), args.negative_sample)
            aux_labels = np.ones(len(batch_aux_data), dtype=np.float32)
            samples, neg_labels, aux_labels = torch.from_numpy(samples), torch.from_numpy(neg_labels), torch.from_numpy(aux_labels)
            # logger.info("Done negative sampling: num negative samples=%d [%.1f s]" % (neg_labels.shape[0], time.time() - ts))

            neg_labels = neg_labels.cuda()
            aux_labels = aux_labels.cuda()

            labels = torch.cat([batch_soft_labels, aux_labels, neg_labels])
            labels = labels.cuda()
            samples = samples.cuda()

            # labels = labels.detach()
            loss = model.get_loss(embed, samples, labels)

            t1 = time.time()

            loss.backward()
            # gpu_usage = get_gpu_memory_map()[int(args.gpu)]
            # logger.info('GPU usage for backward %d' % gpu_usage)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
            optimizer.step()
            t2 = time.time()
            backward_time = t2 - t1
            optimizer.zero_grad()
            print("Training for virtual data: Epoch {:04d} |Batch {:04d} |Loss {:.4f} | Best MRR {:.4f} | | Backward {:.4f}s".
                  format(epoch, batch, loss.item(), best_mrr, backward_time))
            logger.info("Train: Epoch {:04d} |Batch {:04d} |Loss {:.4f} | Best MRR {:.4f} [{:.1f} s]".
                  format(epoch, batch, loss.item(), best_mrr, time.time() - ts))

        # start validate test data
        if epoch % args.evaluate_every == 0:
            logger.info("\n\nValidation...")
            # perform validation on CPU because full graph is too large

            total_data = np.concatenate([train_data, aux_data, virtual_data])

            # build adj list and calculate degrees for sampling for the total data
            total_adj_list, total_degrees = get_adj_and_degrees(num_nodes, total_data)
            total_graph, total_rel, total_norm = build_test_graph(num_nodes, num_rels, total_data)
            total_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
            total_rel = torch.from_numpy(total_rel)
            total_norm = node_norm_to_edge_norm(total_graph, torch.from_numpy(total_norm).view(-1, 1))

            if use_cuda:
                model.cpu()
            model.eval()
            print("start eval")

            total_graph = total_graph.cpu()
            total_node_id = total_node_id.cpu()
            total_rel = total_rel.cpu()
            total_norm = total_norm.cpu()

            embed = model(total_graph, total_node_id, total_rel, total_norm)
            # validation
            mrr_s, mrr_o, mrr, hits_s, hits_o, hits = calc_mrr(embed, model.w_relation, torch.LongTensor(total_data),
                                 torch.LongTensor(valid_data), torch.LongTensor(test_data), hits=[1, 3, 10], eval_bz=args.eval_batch_size,
                                 eval_p=args.eval_protocol)
            logger.info("Validation: Epoch {:04d} | MRR_S {:.4f} | MRR_O {:.4f} | MRR {:.4f} "
                        "| Hits_s(1, 3, 10) {} | Hits_o {} | Hits {} [{:.1f} s]".
                  format(epoch, mrr_s, mrr_o, mrr, hits_s, hits_o, hits, time.time() - ts))

            # save best model
            if best_mrr < mrr:
                best_mrr = mrr

                torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'mrr': best_mrr},
                           model_path)
                logger.info("Better model!: Epoch {:04d} | Best MRR {:.4f} [{:.1f} s]. Save model {}".
                            format(epoch, mrr, time.time() - ts, model_path))

            if use_cuda:
                model.cuda()

            if epoch >= args.epochs:
                print("process of training virtual data done")
                break
    print("training done")
    logger.info("Done training! [%.1f s]" % (time.time() - ts))

    print("\nstart testing:")
    logger.info("\n\n Start testing")
    # use best model checkpoint
    checkpoint = torch.load(model_path)
    if use_cuda:
        model.cpu()  # test on CPU
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    print("Using best epoch: {}".format(checkpoint['epoch']))
    total_graph = total_graph.cpu()
    total_node_id = total_node_id.cpu()
    total_rel = total_rel.cpu()
    total_norm = total_norm.cpu()
    embed = model(total_graph, total_node_id, total_rel, total_norm)
    mrr_s, mrr_o, mrr, hits_s, hits_o, hits = calc_mrr(embed, model.w_relation, torch.LongTensor(total_data), torch.LongTensor(valid_data),
                   torch.LongTensor(test_data), hits=[1, 3, 10], eval_bz=args.eval_batch_size, eval_p=args.eval_protocol)
    logger.info("Done testing.:  MRR_S {:.4f} | MRR_S {:.4f} | MRR_S {:.4f} "
                "| Hits_s(1, 3, 10) {} | Hits_o {} | Hits {} [{:.1f} s]".
                format(mrr_s, mrr_o, mrr, hits_s, hits_o, hits, time.time() - ts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Link prediction for knowledge graphs')

    # my additional parameters
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--eval-batch-size", type=int, default=500,
                        help="batch size when evaluating")
    parser.add_argument("--eval-protocol", type=str, default="filtered",
                        help="type of evaluation protocol: 'raw' or 'filtered' mrr")
    parser.add_argument("--edge-sampler", type=str, default="uniform",
                        help="type of edge sampler: 'uniform' or 'neighbor'")
    parser.add_argument("--n_epochs_aux", type=int, default=None,
                        help="number of epochs to pretrain train aux data")
    parser.add_argument("--isSigmoid", type=bool, default=False,
                        help="whether the score is input into sigmoid function")
    parser.add_argument("--train-graph-size", type=int, default=500000,
                        help="the size of train data used to construct the graph")
    parser.add_argument("--graph-batch-size", type=int,  default=30000,
                        help="the graph size when pretrain the aux data")

    # your parameters
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--n-bases", type=int, default=4)
    parser.add_argument("--regularization", type=float, default=1e-2)
    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--graph-split-size", type=float, default=0.5)

    parser.add_argument("--negative-sample", type=int, default=32)
    parser.add_argument("--evaluate-every", type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128,
                        help='input batch size for testing/validation (default: 128)')
    parser.add_argument('--epochs', type=int, default=5000, help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--data', type=str, default='yago37',
                        help='Dataset to use: {fb15k, wn18, yago37, fb15k237, wn18rr}')
    parser.add_argument('--sub-data', type=str, default='subject-10',
                        help='SUB-Dataset to use: {subject-10...}')
    parser.add_argument('--l2', type=float, default=0.0,
                        help='Weight decay value to use in the optimizer. Default: 0.0')
    parser.add_argument('--model', type=str, default='distmult', help='Choose from: {conve, distmult, complex}')
    parser.add_argument('--embedding-dim', type=int, default=200, help='The embedding dimension (1D). Default: 200')
    parser.add_argument('--hidden-size', type=int, default=9728)
    parser.add_argument('--embedding-shape1', type=int, default=20,
                        help='The first dimension of the reshaped 2D embedding. The second dimension is infered. Default: 20')
    parser.add_argument('--hidden-drop', type=float, default=0.3, help='Dropout for the hidden layer. Default: 0.3.')
    parser.add_argument('--input-drop', type=float, default=0.2, help='Dropout for the input embeddings. Default: 0.2.')
    parser.add_argument('--feat-drop', type=float, default=0.2,
                        help='Dropout for the convolutional features. Default: 0.2.')
    parser.add_argument('--lr-decay', type=float, default=0.995,
                        help='Decay the learning rate by this factor every epoch. Default: 0.995')
    parser.add_argument('--resume', action='store_true', help='Resume a model.')
    parser.add_argument('--use-bias', action='store_true', help='Use a bias in the convolutional layer. Default: True')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing value to use. Default: 0.1')
    parser.add_argument("--penalty", type=float, default=1)
    # ablation study
    parser.add_argument("--hard-label", type=bool, default=False)
    parser.add_argument("--inner-rule", type=bool, default=True)
    parser.add_argument("--iter", type=bool, default=False)
    args = parser.parse_args()
    print(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args)
