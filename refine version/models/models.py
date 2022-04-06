import torch
import torch.nn as nn
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_
from torch_geometric.nn.conv import MessagePassing

from utils.util_dgl_or_pyg import uniform


class Complex(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations, num_bases):
        super(Complex, self).__init__()
        self.num_entities = num_entities
        self.emb_e_real = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel_real = nn.Parameter(torch.Tensor(num_relations, args.embedding_dim))
        self.emb_rel_img = nn.Parameter(torch.Tensor(num_relations, args.embedding_dim))
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.conv1 = RGCNConv(
            args.embedding_dim, args.embedding_dim, num_relations * 2, num_bases=num_bases)
        self.conv2 = RGCNConv(
            args.embedding_dim, args.embedding_dim, num_relations * 2, num_bases=num_bases)
        self.dropout_ratio = args.dropout

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        nn.init.xavier_uniform_(self.emb_rel_real,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.emb_rel_img,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, entity, edge_index, edge_type, edge_norm):
        e1_embedded_real = self.emb_e_real(entity.squeeze())
        e1_embedded_img = self.emb_e_img(entity.squeeze())
        e1_embedded_real = F.relu(self.conv1(e1_embedded_real, edge_index, edge_type))
        e1_embedded_real = F.dropout(e1_embedded_real, p=self.dropout_ratio)
        e1_embedded_real = self.conv2(e1_embedded_real, edge_index, edge_type)

        e1_embedded_img = F.relu(self.conv1(e1_embedded_img, edge_index, edge_type,edge_norm))
        e1_embedded_img = F.dropout(e1_embedded_img, p=self.dropout_ratio)
        e1_embedded_img = self.conv2(e1_embedded_img, edge_index, edge_type, edge_norm)

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
        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = nn.Parameter(torch.Tensor(num_relations, args.embedding_dim))
        self.inp_drop = torch.nn.Dropout(args.input_drop)

        self.conv1 = RGCNConv(
            args.embedding_dim, args.embedding_dim, num_relations * 2, num_bases=num_bases)
        self.conv2 = RGCNConv(
            args.embedding_dim, args.embedding_dim, num_relations * 2, num_bases=num_bases)
        self.dropout_ratio = args.dropout

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        nn.init.xavier_uniform_(self.emb_rel,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, entity, edge_index, edge_type, edge_norm):
        e1_embedded = self.emb_e(entity.squeeze())
        e1_embedded = F.relu(self.conv1(e1_embedded, edge_index, edge_type, edge_norm))
        e1_embedded = F.dropout(e1_embedded, p=self.dropout_ratio)
        e1_embedded = self.conv2(e1_embedded, edge_index, edge_type, edge_norm)

        return e1_embedded

    def get_score(self, triplets, embedding):
        s = embedding[triplets[:, 0]]
        r = self.emb_rel[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        score = s * r * o
        return score

    def score_loss(self, embedding, triplets, target):
        score = self.get_score(triplets, embedding)
        score = torch.sum(score, dim=1)
        score = torch.sigmoid(score)
        return F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding):
        return torch.mean(embedding) + torch.mean(self.emb_rel.pow(2))

class TransE(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations, num_bases):
        super(TransE, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = nn.Parameter(torch.Tensor(num_relations, args.embedding_dim))
        self.inp_drop = torch.nn.Dropout(args.input_drop)


        self.conv1 = RGCNConv(
            args.embedding_dim, args.embedding_dim, num_relations * 2, num_bases=num_bases)
        self.conv2 = RGCNConv(
            args.embedding_dim, args.embedding_dim, num_relations * 2, num_bases=num_bases)
        self.dropout_ratio = args.dropout

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        nn.init.xavier_uniform_(self.emb_rel,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, entity, edge_index, edge_type, edge_norm):
        e1_embedded = self.emb_e(entity.squeeze())
        e1_embedded = F.relu(self.conv1(e1_embedded, edge_index, edge_type, edge_norm))
        e1_embedded = F.dropout(e1_embedded, p=self.dropout_ratio)
        e1_embedded = self.conv2(e1_embedded, edge_index, edge_type, edge_norm)

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
        score = torch.sigmoid(score)
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
        self.conv1 = RGCNConv(
            args.embedding_dim, args.embedding_dim, num_relations * 2, num_bases=num_bases)
        self.conv2 = RGCNConv(
            args.embedding_dim, args.embedding_dim, num_relations * 2, num_bases=num_bases)
        self.dropout_ratio = args.dropout

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        nn.init.xavier_uniform_(self.emb_rel,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, entity, edge_index, edge_type, edge_norm):
        e1_embedded = self.emb_e(entity.squeeze())
        e1_embedded = F.relu(self.conv1(e1_embedded, edge_index, edge_type, edge_norm))
        e1_embedded = F.dropout(e1_embedded, p=self.dropout_ratio)
        e1_embedded = self.conv2(e1_embedded, edge_index, edge_type, edge_norm)

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
        score = torch.sigmoid(score)
        return F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding):
        return torch.mean(embedding) + torch.mean(self.emb_rel.pow(2))


class RGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, **kwargs):
        super(RGCNConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)

    def message(self, x_j, edge_index_j, edge_type, edge_norm):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        # If no node features are given, we implement a simple embedding
        # loopkup based on the target node index and its edge type.
        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)
