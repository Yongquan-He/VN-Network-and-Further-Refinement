import torch
import torch.nn as nn
from dgl.nn import RelGraphConv
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_

class Complex(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations, num_bases):
        super(Complex, self).__init__()
        self.num_entities = num_entities
        self.emb_e_real = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel_real = nn.Parameter(torch.Tensor(num_relations, args.embedding_dim))
        self.emb_rel_img = nn.Parameter(torch.Tensor(num_relations, args.embedding_dim))
        self.conv1 = RelGraphConv(args.embedding_dim, args.embedding_dim, num_relations * 2, "bdd",
                                  num_bases, activation=F.relu, self_loop=True, dropout=args.dropout)
        self.conv2 = RelGraphConv(args.embedding_dim, args.embedding_dim, num_relations * 2, "bdd",
                                  num_bases, activation=None, self_loop=True, dropout=args.dropout)

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        nn.init.xavier_uniform_(self.emb_rel_real,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.emb_rel_img,
                                gain=nn.init.calculate_gain('relu'))


    def forward(self, h1, h2, g, r, norm):
        e1_embedded_real = self.emb_e_real(h1.squeeze())
        e1_embedded_img = self.emb_e_img(h2.squeeze())
        e1_embedded_real = F.relu(self.conv1(g, e1_embedded_real, r, norm))
        e1_embedded_real = self.conv2(g, e1_embedded_real, r, norm)

        e1_embedded_img = F.relu(self.conv1(g, e1_embedded_img, r, norm))
        e1_embedded_img = self.conv2(g, e1_embedded_img, r, norm)

        return e1_embedded_real, e1_embedded_img

    def get_score(self, triplets, embedding_e_real, embedding_e_imag):
        embedding_real = embedding_e_real
        embedding_imag = embedding_e_imag
        relation_embedding_real = self.emb_rel_real
        relation_embedding_imag = self.emb_rel_imag
        s_real = embedding_real[triplets[:, 0]]
        s_img = embedding_imag[triplets[:, 0]]
        o_real = embedding_real[triplets[:, 2]]
        o_img = embedding_imag[triplets[:, 2]]
        r_real = relation_embedding_real[triplets[:, 1]]
        r_img = relation_embedding_imag[triplets[:, 1]]
        score = s_real*r_real*o_real + s_real*r_img*o_img + s_img*r_real*o_img + s_img*r_img*o_real
        return score

    def score_loss(self, embedding_e_real, embedding_e_imag, triplets, target):
        score = self.get_score(triplets, embedding_e_real, embedding_e_imag)
        score = torch.sum(score, dim=1)
        score = torch.sigmoid(score)
        return F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding_real, embedding_imag):
        return torch.mean(embedding_real.pow(2)) + torch.mean(embedding_imag.pow(2)) + torch.mean(self.emb_rel.pow(2))

class DistMult(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations, num_bases):
        super(DistMult, self).__init__()
        # self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_e = nn.Parameter(torch.Tensor(num_entities, args.embedding_dim))
        self.emb_rel = nn.Parameter(torch.Tensor(num_relations, args.embedding_dim))

        self.conv1 = RelGraphConv(args.embedding_dim, args.embedding_dim, num_relations * 2, "bdd",
                num_bases, activation=F.relu, self_loop=True, dropout=args.dropout)
        self.conv2 = RelGraphConv(args.embedding_dim, args.embedding_dim, num_relations * 2, "bdd",
                num_bases, activation=None, self_loop=True, dropout=args.dropout)

    def init(self):
        # xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_e)
        nn.init.xavier_uniform_(self.emb_rel,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, h, g, r, norm):
        # e1_embedded = self.emb_e(h.squeeze())
        # e1_embedded = F.relu(self.conv1(g, e1_embedded, r, norm))
        # e1_embedded = self.conv2(g, e1_embedded, r, norm)

        return self.emb_e

    def get_score(self, triplets, embedding):
        s = embedding[triplets[:, 0]]
        r = self.emb_rel[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        score = s * r * o
        return score

    def score_loss(self, embedding, triplets, target):
        score = self.get_score(triplets, embedding)
        score = torch.sum(score, dim=1)
        return F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding):
        return torch.mean(embedding) + torch.mean(self.emb_rel.pow(2))

class TransE(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations, num_bases):
        super(TransE, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = nn.Parameter(torch.Tensor(num_relations, args.embedding_dim))

        self.conv1 = RelGraphConv(args.embedding_dim, args.embedding_dim, num_relations * 2, "bdd",
                                  num_bases, activation=F.relu, self_loop=True, dropout=args.dropout)
        self.conv2 = RelGraphConv(args.embedding_dim, args.embedding_dim, num_relations * 2, "bdd",
                                  num_bases, activation=None, self_loop=True, dropout=args.dropout)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        nn.init.xavier_uniform_(self.emb_rel,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, h, g, r, norm):
        e1_embedded = self.emb_e(h)
        e1_embedded = F.relu(self.conv1(g, e1_embedded, r, norm))
        e1_embedded = self.conv2(g, e1_embedded, r, norm)

        return e1_embedded

    def get_score(self, triplets, embedding):
        s = embedding[triplets[:, 0]]
        r = self.emb_e[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        score = s + r - o
        return score

    def score_loss(self, embedding, triplets, target):
        score = self.get_score(triplets, embedding)
        score = torch.sum(score, dim=1)
        return F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding):
        return torch.mean(embedding) + torch.mean(self.emb_rel.pow(2))

class ConvE(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations, num_bases):
        super(ConvE, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = nn.Parameter(torch.Tensor(num_relations, args.embedding_dim))
        self.hidden_drop = torch.nn.Dropout(args.hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(args.feat_drop)
        self.emb_dim1 = args.embedding_shape1
        self.emb_dim2 = args.embedding_dim // self.emb_dim1

        self.conv = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=args.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(args.hidden_size, args.embedding_dim)
        self.conv1 = RelGraphConv(args.embedding_dim, args.embedding_dim, num_relations * 2, "bdd",
                                  num_bases, activation=F.relu, self_loop=True, dropout=args.dropout)
        self.conv2 = RelGraphConv(args.embedding_dim, args.embedding_dim, num_relations * 2, "bdd",
                                  num_bases, activation=None, self_loop=True, dropout=args.dropout)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        nn.init.xavier_uniform_(self.emb_rel,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, h, g, r, norm):
        e1_embedded = self.emb_e(h)
        e1_embedded = F.relu(self.conv1(g, e1_embedded, r, norm))
        e1_embedded = self.conv2(g, e1_embedded, r, norm)

        return e1_embedded

    def get_score(self, triplets, embedding):

        s = embedding[triplets[:, 0]]
        r = self.emb_rel[triplets[:, 1]]
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

        return score

    def score_loss(self, embedding, triplets, target):
        score = self.get_score(triplets, embedding)
        score = torch.sum(score, dim=1)
        return F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding):
        return torch.mean(embedding) + torch.mean(self.emb_rel.pow(2))



