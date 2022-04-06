import argparse
import os.path

from tqdm import trange

from models.models import ConvE, DistMult, Complex, TransE
from models.rule_confidence import get_confidence_for_one_hop_rule, get_confidence_for_two_hop_rule
from models.soft_label import *
from utils.util_dgl_or_pyg import *


def train(known_labels, sample_triplets, model, use_cuda, split_size, negative_sample, reg_ratio, num_relations):

    train_data = generate_sampled_graph_and_labels(known_labels, sample_triplets, split_size, num_relations, negative_sample)

    if use_cuda:
        device = torch.device('cuda')
        train_data.to(device)
        model.to(device)

    entity_embedding = model(train_data.entity, train_data.edge_index, train_data.edge_type, train_data.edge_norm)
    loss = model.score_loss(entity_embedding, train_data.samples, train_data.labels) + reg_ratio * model.reg_loss(entity_embedding)

    return loss, train_data.entity, entity_embedding

def get_batches(triples, labels, batch_size):
    n_batches = len(triples) // batch_size

    for ii in range(0, n_batches * batch_size, batch_size):
        if ii != (n_batches - 1) * batch_size:
            X, Y = triples[ii: ii + batch_size], labels[ii: ii + batch_size]
        else:
            X, Y = triples[ii:], labels[ii:]
        yield X, Y

def main(arg):

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

    best_mrr = 0

    target_dir = os.path.join('./data', args.data, args.sub_data)

    model_name = '{2}_{0}_{1}'.format(args.input_drop, args.hidden_drop, args.model)
    model_path = os.path.join("save_models", '{0}_{1}_{2}.model'.format(args.data, args.sub_data, model_name))

    entity2id, relation2id, train_triplets, valid_triplets, aux_triplets, test_triplets = load_data(target_dir)
    triples_set = set()
    triples_set.update(train_triplets)
    triples_set.update(valid_triplets)
    triples_set.update(aux_triplets)
    another_set = set(test_triplets)

    one_hop_pre, one_hop_ground, two_hop_pre1, two_hop_pre2, two_hop_ground = load_vn_data(target_dir, triples_set, another_set)
    # in case of the shape of two hop is 0
    two_hop_pre1.append((1, 1, 1))
    two_hop_pre2.append((1, 1, 1))
    two_hop_ground.append((1, 1, 1))

    train_triplets = np.array(train_triplets)
    aux_triplets = np.array(aux_triplets)
    valid_triplets = np.array(valid_triplets)
    test_triplets = np.array(test_triplets)
    one_hop_pre = np.array(one_hop_pre)
    one_hop_ground = np.array(one_hop_ground)
    two_hop_pre1 = np.array(two_hop_pre1)
    two_hop_pre2 = np.array(two_hop_pre2)
    two_hop_ground = np.array(two_hop_ground)

    all_triplets = torch.LongTensor(np.concatenate((train_triplets, valid_triplets, aux_triplets, test_triplets, one_hop_ground, two_hop_ground)))

    test_triplets = torch.LongTensor(test_triplets)
    train_graph = build_test_graph(len(entity2id), len(relation2id), np.concatenate((train_triplets, aux_triplets, one_hop_ground, two_hop_ground)))

    all_train_triplets = np.concatenate((train_triplets, aux_triplets, one_hop_ground, two_hop_ground))



    if args.model is None:
        model = ConvE(args, len(entity2id), len(relation2id), args.n_bases)
    elif args.model == 'conve':
        model = ConvE(args, len(entity2id), len(relation2id), args.n_bases)
    elif args.model == 'distmult':
        model = DistMult(args, len(entity2id), len(relation2id), args.n_bases)
    elif args.model == 'complex':
        model = Complex(args, len(entity2id), len(relation2id), args.n_bases)
    elif args.model == 'transe':
        model = TransE(args, len(entity2id), len(relation2id), args.n_bases)
    else:
        print('Unknown model: {0}', args.model)
        raise Exception("Unknown model!")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    print(model)

    if use_cuda:
        model.cuda()
    batch_size = args.batch_size
    split_size = args.graph_split_size
    negative_sample = args.negative_sample
    reg_ratio = args.regularization
    for epoch in trange(1, (args.epochs + 1), desc='Epochs', position=0):

        # total_loss = 0
        # one_hop_confidence = get_confidence_for_one_hop_rule(one_hop_pre, one_hop_ground, model, cache)
        # two_hop_confidence = get_confidence_for_two_hop_rule(two_hop_pre1, two_hop_pre2, two_hop_ground, model, cache)

        # positive_labels = torch.ones(len(train_triplets)+len(aux_triplets))
        # one_hop_labels = get_soft_labels_for_one_hop(one_hop_pre, one_hop_ground, one_hop_confidence, args.penalty, model, cache)
        # two_hop_labels = get_soft_labels_for_two_hop(two_hop_pre1, two_hop_pre2, two_hop_ground, two_hop_confidence, args.penalty, model, cache)
        # labels = torch.cat([positive_labels, one_hop_labels, two_hop_labels], dim=0).detach().numpy()
        # labels = torch.ones(len(train_triplets)+len(aux_triplets)+len(one_hop_ground)+len(two_hop_ground))

        '''
        for sample_data, sample_labels in get_batches(all_train_triplets, labels, batch_size):
            model.train()

            optimizer.zero_grad()
            loss, entity, entity_embedding = train(sample_labels, sample_data, model, use_cuda, split_size, negative_sample, reg_ratio, len(relation2id))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)

            optimizer.step()
            total_loss = total_loss + loss
        '''
        train_data = generate_sampled_graph_and_labels_v1(all_train_triplets, batch_size, split_size, len(entity2id),
                                                       len(relation2id), negative_sample)

        if use_cuda:
            device = torch.device('cuda')
            train_data.to(device)

        entity_embedding = model(train_data.entity, train_data.edge_index, train_data.edge_type, train_data.edge_norm)
        loss = model.score_loss(entity_embedding, train_data.samples, train_data.labels) + reg_ratio * model.reg_loss(
            entity_embedding)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)

        optimizer.step()
        if epoch % args.evaluate_every == 0:

            tqdm.write("Train Loss {} at epoch {}".format(loss, epoch))

            if use_cuda:
                model.cpu()

            model.eval()
            cache = model(train_graph.entity, train_graph.edge_index, train_graph.edge_type,
                                     train_graph.edge_norm)
            test_mrr = calc_mrr(cache, model.emb_rel, test_triplets, all_triplets, hits=[1, 3, 10])
            
            if test_mrr > best_mrr:
                best_mrr = test_mrr
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_path)

            if use_cuda:
                model.cuda()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Link prediction for knowledge graphs')
    # RGCN
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--n-bases", type=int, default=4)
    parser.add_argument("--regularization", type=float, default=1e-2)
    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--graph-split-size", type=float, default=0.5)

    parser.add_argument("--negative-sample", type=int, default=64)
    parser.add_argument("--evaluate-every", type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=5000, help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=5000, help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.02, help='learning rate (default: 0.003)')
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--data', type=str, default='wn18rr',
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

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    main(args)

    args = parser.parse_args()
    print(args)