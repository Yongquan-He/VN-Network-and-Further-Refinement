import torch
import math
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_
from config import config
import torch.nn as nn
Config = config()
FloatTensor = torch.cuda.FloatTensor

class Complex(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(Complex, self).__init__()
        self.num_entities = num_entities
        self.emb_e_real = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1, rel):

        e1_embedded_real = self.emb_e_real(e1).squeeze()
        rel_embedded_real = self.emb_rel_real(rel).squeeze()
        e1_embedded_img =  self.emb_e_img(e1).squeeze()
        rel_embedded_img = self.emb_rel_img(rel).squeeze()

        e1_embedded_real = self.inp_drop(e1_embedded_real)
        rel_embedded_real = self.inp_drop(rel_embedded_real)
        e1_embedded_img = self.inp_drop(e1_embedded_img)
        rel_embedded_img = self.inp_drop(rel_embedded_img)

        # complex space bilinear product (equivalent to HolE)
        realrealreal = torch.mm(e1_embedded_real*rel_embedded_real, self.emb_e_real.weight.transpose(1,0))
        realimgimg = torch.mm(e1_embedded_real*rel_embedded_img, self.emb_e_img.weight.transpose(1,0))
        imgrealimg = torch.mm(e1_embedded_img*rel_embedded_real, self.emb_e_img.weight.transpose(1,0))
        imgimgreal = torch.mm(e1_embedded_img*rel_embedded_img, self.emb_e_real.weight.transpose(1,0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = F.sigmoid(pred)

        return pred

class DistMult(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(DistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1)
        rel_embedded= self.emb_rel(rel)
        e1_embedded = e1_embedded.squeeze()
        rel_embedded = rel_embedded.squeeze()

        e1_embedded = self.inp_drop(e1_embedded)
        rel_embedded = self.inp_drop(rel_embedded)

        pred = torch.mm(e1_embedded*rel_embedded, self.emb_e.weight.transpose(1,0))
        pred = F.sigmoid(pred)

        return pred

class ConvE(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(ConvE, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(Config.feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=Config.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(10368,Config.embedding_dim)
        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        batch_size = e1.size(0)
        e1_embedded= self.emb_e(e1).view(-1, 1, 10, 20)
        rel_embedded = self.emb_rel(rel).view(-1, 1, 10, 20)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1,0)) # shape (batch, n_ent)
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)

        return pred

class ConvTransE(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(ConvTransE, self).__init__()

        self.emb_e = torch.nn.Embedding(num_entities, Config.init_emb_size, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.init_emb_size, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout_rate)
        self.feature_map_drop = torch.nn.Dropout(Config.dropout_rate)
        self.loss = torch.nn.BCELoss()

        self.conv1 = nn.Conv1d(2, Config.channels, Config.kernel_size, stride=1, padding=int(
            math.floor(Config.kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(Config.channels)
        self.bn2 = torch.nn.BatchNorm1d(Config.init_emb_size)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(Config.init_emb_size * Config.channels, Config.init_emb_size)
        # self.bn3 = torch.nn.BatchNorm1d(Config.gc1_emb_size)
        # self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(Config.init_emb_size)

        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel, X, A):
        emb_initial = self.emb_e(X)
        e1_embedded_all = self.bn_init(emb_initial)
        e1_embedded = e1_embedded_all[e1]
        rel_embedded = self.emb_rel(rel)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(Config.batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        pred = F.sigmoid(x)

        return pred

class SACN(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(SACN, self).__init__()

        self.emb_e = torch.nn.Embedding(num_entities, Config.init_emb_size, padding_idx=0)
        self.gc1 = GraphConvolution(Config.init_emb_size, Config.gc1_emb_size, num_relations)
        self.gc2 = GraphConvolution(Config.gc1_emb_size, Config.embedding_dim, num_relations)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout_rate)
        self.feature_map_drop = torch.nn.Dropout(Config.dropout_rate)
        self.loss = torch.nn.BCELoss()
        self.conv1 = nn.Conv1d(2, Config.channels, Config.kernel_size, stride=1, padding=int(
            math.floor(Config.kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(Config.channels)
        self.bn2 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(Config.embedding_dim * Config.channels, Config.embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(Config.gc1_emb_size)
        self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(Config.init_emb_size)

        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)
        xavier_normal_(self.gc1.weight.data)
        xavier_normal_(self.gc2.weight.data)

    def forward(self, e1, rel, X, A):
        emb_initial = self.emb_e(X)
        x = self.gc1(emb_initial, A)
        x = self.bn3(x)
        x = F.tanh(x)
        x = F.dropout(x, Config.dropout_rate, training=self.training)

        x = self.bn4(self.gc2(x, A))
        e1_embedded_all = F.tanh(x)
        e1_embedded_all = F.dropout(e1_embedded_all, Config.dropout_rate, training=self.training)
        e1_embedded = e1_embedded_all[e1]
        rel_embedded = self.emb_rel(rel)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(Config.batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        pred = F.sigmoid(x)

        return pred

class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, num_relations, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.num_relations = num_relations
        self.alpha = torch.nn.Embedding(num_relations + 1, 1, padding_idx=0)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        alp = self.alpha(adj[1]).t()[0]
        A = torch.sparse_coo_tensor(adj[0], alp, torch.Size([adj[2], adj[2]]), requires_grad=True)
        A = A + A.transpose(0, 1)
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(A, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class VN_WGCN(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(VN_WGCN, self).__init__()

        self.emb_e = torch.nn.Embedding(num_entities, Config.init_emb_size, padding_idx=0)
        self.gc1 = GraphConvolution(Config.init_emb_size, Config.gc1_emb_size, num_relations)
        self.gc2 = GraphConvolution(Config.gc1_emb_size, Config.gc1_emb_size, num_relations)
        self.gc3 = GraphConvolution(Config.gc1_emb_size, Config.embedding_dim, num_relations)
        self.query = QueryLayer(num_relations, num_entities, Config.gc1_emb_size, Config.embedding_dim, concat=True)

        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout_rate)
        self.loss = torch.nn.BCELoss()
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.bn1 = torch.nn.BatchNorm1d(Config.gc1_emb_size)
        self.bn2 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(Config.init_emb_size)
        self.num_entities = num_entities
        print(num_entities, num_relations)
        '''新加的接口'''
        self.embedding_final = torch.FloatTensor(num_entities, Config.embedding_dim)
        self.query_embedding_final = torch.FloatTensor(num_relations, num_entities, Config.embedding_dim)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)
        xavier_normal_(self.gc1.weight.data)
        xavier_normal_(self.gc2.weight.data)

    def forward(self, e1, rel, e2, X, A):
        emb_initial = self.emb_e(X)
        x = self.gc1(emb_initial, A)
        x = self.bn1(x)
        x = F.tanh(x)
        x = F.dropout(x, Config.dropout_rate, training=self.training)
        x = self.bn1(self.gc2(x, A))
        x = F.tanh(x)
        x = F.dropout(x, Config.dropout_rate, training=self.training)
        if Config.use_query is False:
            x = self.bn2(self.gc3(x, A))
            e1_embedded_all = F.tanh(x)
            e1_embedded_all = F.dropout(e1_embedded_all, Config.dropout_rate, training=self.training)
            e1_embedded = e1_embedded_all[e1]
            e2_embedded = e1_embedded_all[e2]
            e1_embedded = e1_embedded.squeeze()
            e2_embedded = e2_embedded.squeeze()
            self.embedding_final = e1_embedded_all
        else:
            x = self.bn2(self.gc3(x, A))
            e1_embedded_all = F.tanh(x)
            e1_embedded_all = F.dropout(e1_embedded_all, Config.dropout_rate, training=self.training)
            e1_embedded = e1_embedded_all[e1]
            e2_embedded = e1_embedded_all[e2]
            e1_embedded = e1_embedded.squeeze()
            e2_embedded = e2_embedded.squeeze()
            self.embedding_final = e1_embedded_all

            
        '''新加的接口'''
        rel_embedded = self.emb_rel(rel)
        rel_embedded = rel_embedded.squeeze()
        x = e1_embedded * rel_embedded * e2_embedded
        x = torch.sum(x, dim=1)
        pred = F.sigmoid(x)
        return pred

    def get_triple_score(self, e1, rel, e2):
        e1_emb = self.embedding_final[e1].squeeze()
        e2_emb = self.embedding_final[e2].squeeze()
        rel_emb = self.emb_rel(rel).squeeze()
        score = F.sigmoid(e1_emb * rel_emb * e2_emb)
        score = torch.sum(score, dim=1)
        return score

    def compute_rank(self, e1, rel, X, A):
        if Config.use_query is False:
            e1_embedded_all = self.embedding_final
            e1_embedded = e1_embedded_all[e1]
            e1_embedded_all = e1_embedded_all.squeeze()
            e1_embedded = e1_embedded.squeeze()
        else:
            e1_embedded_all = self.embedding_final
            e1_embedded_query = self.query_embedding_final
            e1_embedded = e1_embedded_query[rel, e1]
            e1_embedded = e1_embedded.squeeze()
            e1_embedded_all = e1_embedded_all.squeeze()
        rel_embedded = self.emb_rel(rel)
        rel_embedded = rel_embedded.squeeze()
        x = torch.mm(e1_embedded * rel_embedded, e1_embedded_all.transpose(1, 0))
        pred = F.sigmoid(x)
        return pred

class VN_WGCN_TransE(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(VN_WGCN, self).__init__()

        self.emb_e = torch.nn.Embedding(num_entities, Config.init_emb_size, padding_idx=0)
        self.gc1 = GraphConvolution(Config.init_emb_size, Config.gc1_emb_size, num_relations)
        self.gc2 = GraphConvolution(Config.gc1_emb_size, Config.gc1_emb_size, num_relations)
        self.gc3 = GraphConvolution(Config.gc1_emb_size, Config.embedding_dim, num_relations)

        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout_rate)
        self.loss = torch.nn.BCELoss()
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.bn1 = torch.nn.BatchNorm1d(Config.gc1_emb_size)
        self.bn2 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(Config.init_emb_size)
        self.num_entities = num_entities
        print(num_entities, num_relations)
        '''新加的接口'''
        self.embedding_final = torch.FloatTensor(num_entities, Config.embedding_dim)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)
        xavier_normal_(self.gc1.weight.data)
        xavier_normal_(self.gc2.weight.data)

    def forward(self, e1, rel, e2, X, A):
        emb_initial = self.emb_e(X)
        x = self.gc1(emb_initial, A)
        x = self.bn1(x)
        x = F.tanh(x)
        x = F.dropout(x, Config.dropout_rate, training=self.training)
        x = self.bn1(self.gc2(x, A))
        x = F.tanh(x)
        x = F.dropout(x, Config.dropout_rate, training=self.training)
        x = self.bn2(self.gc3(x, A))

        e1_embedded_all = F.tanh(x)
        e1_embedded_all = F.dropout(e1_embedded_all, Config.dropout_rate, training=self.training)
        '''新加的接口'''
        self.embedding_final = e1_embedded_all
        e1_embedded = e1_embedded_all[e1]
        e2_embedded = e1_embedded_all[e2]
        rel_embedded = self.emb_rel(rel)
        e1_embedded = e1_embedded.squeeze()
        rel_embedded = rel_embedded.squeeze()
        e2_embedded = e2_embedded.squeeze()
        x = e1_embedded + rel_embedded - e2_embedded
        pred = torch.norm(x, p=2, dim=1)
        return pred

    def get_triple_score(self, e1, rel, e2):
        e1_emb = self.embedding_final[e1].squeeze()
        e2_emb = self.embedding_final[e2].squeeze()
        rel_emb = self.emb_rel(rel).squeeze()
        score = e1_emb + rel_emb - e2_emb
        score = torch.norm(score, p=2, dim=1)
        return score

    def compute_rank(self, e1, rel, X, A):
        e1_embedded_all = self.embedding_final
        e1_embedded = e1_embedded_all[e1]
        rel_embedded = self.emb_rel(rel)
        e1_embedded_all = e1_embedded_all.squeeze().view(1, self.num_entities*Config.embedding_dim).repeat(e1.size, 1)
        rel_embedded = rel_embedded.squeeze()
        e1_embedded = e1_embedded.squeeze()
        e1_embedded = e1_embedded.repeat(1, self.num_entities)
        rel_embedded = rel_embedded.repeat(1, self.num_entities)
        x = e1_embedded + rel_embedded - e1_embedded_all
        x = x.view(e1.size, self.num_entities, Config.embedding_dim)
        x = torch.norm(x, p=2, dim=2)
        pred = x.squeeze()
        return pred

class VN_WGCN_Complex(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(VN_WGCN_Complex, self).__init__()

        self.emb_e_real = torch.nn.Embedding(num_entities, Config.init_emb_size, padding_idx=0)
        self.gc1_real = GraphConvolution(Config.init_emb_size, Config.gc1_emb_size, num_relations)
        self.gc2_real = GraphConvolution(Config.gc1_emb_size, Config.gc1_emb_size, num_relations)
        self.gc3_real = GraphConvolution(Config.gc1_emb_size, Config.embedding_dim, num_relations)

        self.emb_e_imag = torch.nn.Embedding(num_entities, Config.init_emb_size, padding_idx=0)
        self.gc1_imag = GraphConvolution(Config.init_emb_size, Config.gc1_emb_size, num_relations)
        self.gc2_imag = GraphConvolution(Config.gc1_emb_size, Config.gc1_emb_size, num_relations)
        self.gc3_img = GraphConvolution(Config.gc1_emb_size, Config.embedding_dim, num_relations)

        self.emb_rel_real = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.emb_rel_imag = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)

        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout_rate)
        self.loss = torch.nn.BCELoss()
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.bn1 = torch.nn.BatchNorm1d(Config.gc1_emb_size)
        self.bn2 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(Config.init_emb_size)
        self.num_entities = num_entities
        print(num_entities, num_relations)
        '''新加的接口'''
        self.embedding_final_real = FloatTensor(num_entities, Config.embedding_dim)
        self.embedding_final_imag = FloatTensor(num_entities, Config.embedding_dim)

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.gc1_real.weight.data)
        xavier_normal_(self.gc2_real.weight.data)
        xavier_normal_(self.gc3_real.weight.data)

        xavier_normal_(self.emb_e_imag.weight.data)
        xavier_normal_(self.emb_rel_imag.weight.data)
        xavier_normal_(self.gc1_imag.weight.data)
        xavier_normal_(self.gc2_imag.weight.data)
        xavier_normal_(self.gc3_img.weight.data)

    def forward(self, e1, rel, e2, X, A):
        emb_initial_real = self.emb_e_real(X)
        x = self.gc1_real(emb_initial_real, A)
        x = self.bn1(x)
        x = F.tanh(x)
        x = F.dropout(x, Config.dropout_rate, training=self.training)
        x = self.bn1(self.gc2_real(x, A))
        x = F.tanh(x)
        x = F.dropout(x, Config.dropout_rate, training=self.training)
        x = self.bn2(self.gc3_real(x, A))

        emb_initial_imag = self.emb_e_imag(X)
        x1 = self.gc1_imag(emb_initial_imag, A)
        x1 = self.bn1(x1)
        x1 = F.tanh(x1)
        x1 = F.dropout(x1, Config.dropout_rate, training=self.training)
        x1 = self.bn1(self.gc2_imag(x1, A))
        x1 = F.tanh(x1)
        x1 = F.dropout(x1, Config.dropout_rate, training=self.training)
        x1 = self.bn2(self.gc3_img(x1, A))

        e1_embedded_all_real = F.tanh(x)
        e1_embedded_all_real = F.dropout(e1_embedded_all_real, Config.dropout_rate, training=self.training)
        e1_embedded_all_imag = F.tanh(x1)
        e1_embedded_all_imag = F.dropout(e1_embedded_all_imag, Config.dropout_rate, training=self.training)
        '''新加的接口'''
        self.embedding_final_real = e1_embedded_all_real
        self.embedding_final_imag = e1_embedded_all_imag
        e1_embedded_real = e1_embedded_all_real[e1].squeeze()
        e2_embedded_real = e1_embedded_all_real[e2].squeeze()
        e1_embedded_imag = e1_embedded_all_imag[e1].squeeze()
        e2_embedded_imag = e1_embedded_all_imag[e2].squeeze()
        rel_embedded_real = self.emb_rel_real(rel).squeeze()
        rel_embedded_imag = self.emb_rel_imag(rel).squeeze()
        realrealreal = e1_embedded_real*rel_embedded_real*e2_embedded_real
        realimagimag = e1_embedded_real*rel_embedded_imag*e2_embedded_imag
        imagrealimag = e1_embedded_imag*rel_embedded_real*e2_embedded_imag
        imagimagreal = e1_embedded_imag*rel_embedded_imag*e2_embedded_real
        pred = realrealreal + realimagimag + imagrealimag - imagimagreal
        pred = torch.sum(pred, dim=1)
        pred = torch.sigmoid(pred)
        return pred

    def get_triple_score(self, e1, rel, e2):
        e1_embedded_real = self.embedding_final_real[e1].squeeze()
        e2_embedded_real = self.embedding_final_real[e2].squeeze()
        e1_embedded_imag = self.embedding_final_imag[e1].squeeze()
        e2_embedded_imag = self.embedding_final_imag[e2].squeeze()
        rel_embedded_real = self.emb_rel_real(rel).squeeze()
        rel_embedded_imag = self.emb_rel_imag(rel).squeeze()
        realrealreal = e1_embedded_real * rel_embedded_real * e2_embedded_real
        realimagimag = e1_embedded_real * rel_embedded_imag * e2_embedded_imag
        imagrealimag = e1_embedded_imag * rel_embedded_real * e2_embedded_imag
        imagimagreal = e1_embedded_imag * rel_embedded_imag * e2_embedded_real
        pred = realrealreal + realimagimag + imagrealimag - imagimagreal
        pred = torch.sum(pred, dim=1)
        pred = torch.sigmoid(pred)
        return pred

    def compute_rank(self, e1, rel, X, A):
        emb_initial_real = self.emb_e_real(X)
        x = self.gc1_real(emb_initial_real, A)
        x = self.bn1(x)
        x = F.tanh(x)
        x = F.dropout(x, Config.dropout_rate, training=self.training)
        x = self.bn1(self.gc2_real(x, A))
        x = F.tanh(x)
        x = F.dropout(x, Config.dropout_rate, training=self.training)
        x = self.bn2(self.gc3_real(x, A))

        emb_initial_imag = self.emb_e_imag(X)
        x1 = self.gc1_imag(emb_initial_imag, A)
        x1 = self.bn1(x1)
        x1 = F.tanh(x1)
        x1 = F.dropout(x1, Config.dropout_rate, training=self.training)
        x1 = self.bn1(self.gc2_imag(x1, A))
        x1 = F.tanh(x1)
        x1 = F.dropout(x1, Config.dropout_rate, training=self.training)
        x1 = self.bn2(self.gc3_img(x1, A))

        e1_embedded_all_real = F.tanh(x)
        e1_embedded_all_real = F.dropout(e1_embedded_all_real, Config.dropout_rate, training=self.training)
        e1_embedded_all_imag = F.tanh(x1)
        e1_embedded_all_imag = F.dropout(e1_embedded_all_imag, Config.dropout_rate, training=self.training)
        e1_embedded_real = e1_embedded_all_real[e1].squeeze()
        e1_embedded_img = e1_embedded_all_imag[e1].squeeze()
        rel_embedded_real = self.emb_rel_real(rel).squeeze()
        rel_embedded_img = self.emb_rel_imag(rel).squeeze()
        realrealreal = torch.mm(e1_embedded_real * rel_embedded_real, self.emb_e_real.weight.transpose(1, 0))
        realimgimg = torch.mm(e1_embedded_real * rel_embedded_img, self.emb_e_img.weight.transpose(1, 0))
        imgrealimg = torch.mm(e1_embedded_img * rel_embedded_real, self.emb_e_img.weight.transpose(1, 0))
        imgimgreal = torch.mm(e1_embedded_img * rel_embedded_img, self.emb_e_real.weight.transpose(1, 0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = torch.sigmoid(pred)
        return pred

class QueryLayer(nn.Module):
    def __init__(self, num_rel, num_e, in_features, out_features, concat=True):
        super(QueryLayer, self).__init__()
        self.num_rel = num_rel
        self.num_e =num_e
        self.dropout = Config.dropout_rate
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(Config.alpha)

    def forward(self, input, adj):
        '''
        :param input: 输入特征 (batch,in_features)
        :param adj:  邻接矩阵 (batch,batch)
        :return: 输出特征 (batch,out_features)
        '''
        h = torch.mm(input, self.W) # (batch,out_features)
        N = h.size()[0] # batch
        a_input = torch.cat([h.repeat(1, N)   # (batch,out_features*batch)
                            .view(N * N, -1), # (batch*batch,out_features)
                             h.repeat(N, 1)], # (batch*batch,out_features)
                            dim=1).view(N, -1, 2 * self.out_features) # (batch,batch,2 * out_features)
        # 通过刚刚的矩阵与权重矩阵a相乘计算每两个样本之间的相关性权重，最后再根据邻接矩阵置零没有连接的权重
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2)) # (batch,batch)
        # 置零的mask
        zero_vec = -9e15*torch.ones_like(e) # (batch,batch)
        '''之前的
        attention = torch.where(adj > 0, e, zero_vec) # (batch,batch) 有相邻就为e位置的值，不相邻则为0
        attention = F.softmax(attention, dim=1)  # (batch,batch)
        attention = F.dropout(attention, self.dropout, training=self.training) # (batch,batch)
        h_prime = torch.matmul(attention, h) # (batch,out_features)
        '''
        attention =adj * e
        attention = attention.repeat(self.num_rel, 1).view(self.num_rel, self.num_e, self.num_e)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)  # (batch, batch, out_features)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


'''我的模板'''
class MyModel(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(DistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1)
        rel_embedded= self.emb_rel(rel)

        # Add your model function here
        # The model function should operate on the embeddings e1 and rel
        # and output scores for all entities (you will need a projection layer
        # with output size num_relations (from constructor above)

        # generate output scores here
        prediction = F.sigmoid(Config.output)

        return prediction


