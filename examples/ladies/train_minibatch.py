import gs
from gs.jit.passes import dce
from gs.utils import SeedGenerator, load_reddit, ConvModel, GraphConv
import torch
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.function as fn
import numpy as np
import time
import argparse

device = torch.device('cuda')
time_list = []


def normalized_laplacian_edata(g, weight=None):
    with g.local_scope():
        if weight is None:
            weight = 'W'
            g.edata[weight] = torch.ones(g.number_of_edges(), device=g.device)
        g_rev = dgl.reverse(g, copy_edata=True)
        g.update_all(fn.copy_e(weight, weight), fn.sum(weight, 'v'))
        g_rev.update_all(fn.copy_e(weight, weight), fn.sum(weight, 'u'))
        g.ndata['u'] = g_rev.ndata['u']
        g.apply_edges(lambda edges: {
            'w':
            edges.data[weight] / torch.sqrt(edges.src['u'] * edges.dst['v'])
        })
        return g.edata['w']


def normalized_edata(g, weight=None):
    with g.local_scope():
        if weight is None:
            weight = 'W'
            g.edata[weight] = torch.ones(g.number_of_edges(), device=g.device)
        g.update_all(fn.copy_e(weight, weight), fn.sum(weight, 'v'))
        g.apply_edges(lambda edges: {'w': 1 / edges.dst['v']})
        return g.edata['w']


def ladies_sampler(P: gs.Matrix, seeds: torch.Tensor, fanouts: list):
    output_node = seeds
    ret = []
    # D_in = A.sum(axis=0)
    # D_out = A.sum(axis=1)
    # P = A.divide(D_out.sqrt(), axis=1).divide(D_in.sqrt(), axis=0)
    for fanout in fanouts:
        U = P[:, seeds]
        prob = U.sum(axis=1, powk=2)
        selected, _ = torch.ops.gs_ops.list_sampling_with_probs(
            U.row_ids(unique=False), prob, fanout, False)
        nodes = torch.cat((seeds, selected)).unique()  # add self-loop
        subU = U[nodes, :].divide(prob[nodes], axis=1).normalize(axis=0)
        seeds = subU.all_indices()
        ret.insert(0, subU.to_dgl_block())
    input_node = seeds
    return input_node, output_node, ret


def evaluate(model, matrix, compiled_func, seedloader, features, labels,
             fanouts):
    model.eval()
    ys = []
    y_hats = []
    for it, seeds in enumerate(seedloader):
        input_nodes, output_nodes, blocks = compiled_func(
            matrix, seeds, fanouts)
        with torch.no_grad():
            x = features[input_nodes]
            y = labels[output_nodes]
            ys.append(y)
            y_hats.append(model(blocks, x))
    return MF.accuracy(torch.cat(y_hats), torch.cat(ys))


def layerwise_infer(graph, nid, model, batch_size, feat, label, edge_weight):
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size, feat,
                               edge_weight)  # pred in buffer_device
        pred = pred[nid]
        label = label[nid].to(pred.device)
        return MF.accuracy(pred, label)


def train(g, dataset, feat_device):
    features, labels, n_classes, train_idx, val_idx, test_idx = dataset
    model = ConvModel(features.shape[1], 256, n_classes, feat_device,
                      GraphConv).to(device)
    # compute edge weight
    g.edata['weight'] = normalized_laplacian_edata(g)
    # create sampler & dataloader
    m = gs.Matrix(gs.Graph(False))
    m.load_dgl_graph(g, 'weight')
    print("Check load successfully:", m._graph._CAPI_metadata(), '\n')
    fanouts = [2000, 2000]
    # compiled_func = gs.jit.compile(
    #     func=ladies_sampler, args=(m, torch.Tensor(), torch.Tensor(), fanouts))
    # compiled_func.gm = dce(compiled_func.gm)
    compiled_func = ladies_sampler
    train_seedloader = SeedGenerator(train_idx,
                                     batch_size=1024,
                                     shuffle=True,
                                     drop_last=False)
    val_seedloader = SeedGenerator(val_idx,
                                   batch_size=1024,
                                   shuffle=True,
                                   drop_last=False)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    n_epoch = 10

    for epoch in range(n_epoch):
        torch.cuda.synchronize()
        start = time.time()

        model.train()
        total_loss = 0
        for it, seeds in enumerate(train_seedloader):
            input_nodes, output_nodes, blocks = compiled_func(
                m, seeds, fanouts)
            x = features[input_nodes]
            y = labels[output_nodes]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        torch.cuda.synchronize()
        time_list.append(time.time() - start)

        acc = evaluate(model, m, compiled_func, val_seedloader, features,
                       labels, fanouts)
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
            epoch, total_loss / (it + 1), acc.item()))
        print(torch.cuda.max_memory_reserved() / (1024 * 1024 * 1024), 'GB')

    print('Average epoch time:', np.mean(time_list[3:]))

    print('Testing...')
    acc = layerwise_infer(g,
                          test_idx,
                          model,
                          batch_size=4096,
                          feat=features,
                          label=labels,
                          edge_weight=g.edata['weight'])
    print("Test Accuracy {:.4f}".format(acc.item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmode",
                        default='cuda',
                        choices=['cpu', 'cuda'],
                        help="Feature reside device. To cpu or gpu")
    args = parser.parse_args()
    print(args)
    feat_device = args.fmode
    # load and preprocess dataset
    print('Loading data')
    g, features, labels, n_classes, splitted_idx = load_reddit()
    print('num of nodes:', g.num_nodes())
    print('num of edges:', g.num_edges())
    g = g.long().to('cuda')
    train_mask, val_mask, test_mask = splitted_idx['train'], splitted_idx[
        'valid'], splitted_idx['test']
    train_idx = train_mask.to(device)
    val_idx = val_mask.to(device)
    features = features.to(feat_device)
    labels = labels.to(device)

    train(g, (features, labels, n_classes, train_idx, val_idx, test_mask),
          feat_device)
