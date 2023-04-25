import gs
from gs.utils import SeedGenerator, load_reddit, load_ogbn_products
import torch
import torch.nn.functional as F
from torch.nn.functional import normalize, relu
from dgl.utils import gather_pinned_tensor_rows
import numpy as np
import time
import tqdm
import argparse
from model import *


def asgcn_matrix_sampler(A: gs.Matrix, seeds, features, W, fanouts, use_uva):
    output_nodes = seeds
    blocks = []
    for fanout in fanouts:
        subA = A[:, seeds]
        p = subA.sum(axis=1, powk=2).sqrt()
        row_indices = subA.row_ids()
        if use_uva:
            node_feats_u = gather_pinned_tensor_rows(features, row_indices)
            node_feats_v = gather_pinned_tensor_rows(features, seeds)
        else:
            node_feats_u = features[row_indices]
            node_feats_v = features[seeds]
        h_u = node_feats_u @ W[:, 0]
        h_v = node_feats_v @ W[:, 1]
        h_v_sum = torch.sum(h_v)
        attention = torch.flatten((relu(h_u + h_v_sum) + 1) / fanout)
        g_u = torch.flatten(relu(h_u) + 1)

        q = normalize(p[row_indices] * attention * g_u, p=1.0, dim=0)

        selected, idx = torch.ops.gs_ops.list_sampling_with_probs(
            row_indices, q, fanout, False)

        subA = subA[selected, :]
        W_tilde = gs.ops.u_add_v(subA, h_u[idx], h_v)
        W_tilde = (relu(W_tilde) + 1) / selected.numel()
        W_tilde = gs.ops.e_div_u(subA, W_tilde, q[idx])
        subA.set_data(W_tilde * subA.get_data())
        u_sum = subA.sum(axis=1)
        u_all = torch.zeros(
            A.get_num_rows(), dtype=torch.float32, device='cuda')
        u_all[selected] = u_sum

        block = subA.to_dgl_block()
        block.srcdata['u_sum'] = u_all[block.srcdata['_ID']]
        seeds = block.srcdata['_ID']
        blocks.insert(0, block)
    input_nodes = seeds
    return input_nodes, output_nodes, blocks


def compute_acc(pred, label):
    return (pred.argmax(1) == label).float().mean()


def train(dataset, args):
    device = args.device
    use_uva = args.use_uva
    fanouts = [int(x.strip()) for x in args.samples.split(',')]

    g, features, labels, n_classes, splitted_idx = dataset
    train_nid, val_nid, _ = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    g, train_nid, val_nid = g.to(device), train_nid.to(
        device), val_nid.to(device)
    adj_weight = normalized_laplacian_edata(g)
    csc_indptr, csc_indices, edge_ids = g.adj_sparse('csc')
    adj_weight = adj_weight[edge_ids]
    if use_uva and device == 'cpu':
        features, labels = features.pin_memory(), labels.pin_memory()
        csc_indptr = csc_indptr.pin_memory()
        csc_indices = csc_indices.pin_memory()
        train_nid, val_nid = train_nid.pin_memory(), val_nid.pin_memory()
        adj_weight = adj_weight.cuda()
    else:
        features, labels = features.to(device), labels.to(device)
        adj_weight = adj_weight.to(device)
    m = gs.Matrix(gs.Graph(False))
    m._graph._CAPI_load_csc(csc_indptr, csc_indices)
    m._graph._CAPI_set_data(adj_weight)
    print("Check load successfully:", m._graph._CAPI_metadata(), '\n')

    # compiled_func = gs.jit.compile(
    #     func=fastgcn_sampler, args=(m, torch.Tensor(), fanouts))
    # compiled_func.gm = dce(slicing_and_sampling_fuse(compiled_func.gm))
    compiled_func = asgcn_matrix_sampler
    train_seedloader = SeedGenerator(
        train_nid, batch_size=args.batchsize, shuffle=True, drop_last=False)
    val_seedloader = SeedGenerator(
        val_nid, batch_size=args.batchsize, shuffle=True, drop_last=False)
    model = Model(features.shape[1], 64, n_classes, len(fanouts)).to('cuda')
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    n_epoch = 5

    sample_time_list = []
    epoch_time = []
    mem_list = []
    feature_loading_list = []
    torch.cuda.synchronize()
    static_memory = torch.cuda.memory_allocated()
    print('memory allocated before training:',
          static_memory / (1024 * 1024 * 1024), 'GB')
    for epoch in range(n_epoch):
        epoch_feature_loading = 0
        sample_time = 0
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()

        model.train()
        total_loss = 0
        torch.cuda.synchronize()
        tic = time.time()
        for it, seeds in enumerate(tqdm.tqdm(train_seedloader)):
            seeds = seeds.to('cuda')
            input_nodes, output_nodes, blocks = compiled_func(
                m, seeds, features, model.W_g, fanouts, use_uva)
            torch.cuda.synchronize()
            sample_time += time.time() - tic

            tic = time.time()
            blocks = [block.to('cuda') for block in blocks]
            if use_uva:
                batch_inputs = gather_pinned_tensor_rows(
                    features, input_nodes)
                batch_labels = gather_pinned_tensor_rows(labels, seeds)
            else:
                batch_inputs = features[input_nodes].to('cuda')
                batch_labels = labels[seeds].to('cuda')
            torch.cuda.synchronize()
            epoch_feature_loading += time.time() - tic

            batch_pred, reg_loss = model(blocks, batch_inputs)
            is_labeled = batch_labels == batch_labels
            batch_labels = batch_labels[is_labeled].long()
            batch_pred = batch_pred[is_labeled]
            loss = F.cross_entropy(batch_pred, batch_labels) + 0.5 * reg_loss
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            torch.cuda.synchronize()
            tic = time.time()

        model.eval()
        val_pred = []
        val_labels = []
        with torch.no_grad():
            torch.cuda.synchronize()
            tic = time.time()
            for it, seeds in enumerate(tqdm.tqdm(val_seedloader)):
                seeds = seeds.to('cuda')
                input_nodes, output_nodes, blocks = compiled_func(
                    m, seeds, features, model.W_g, fanouts, use_uva)
                torch.cuda.synchronize()
                sample_time += time.time() - tic

                tic = time.time()
                blocks = [block.to('cuda') for block in blocks]
                if use_uva:
                    batch_inputs = gather_pinned_tensor_rows(
                        features, input_nodes)
                    batch_labels = gather_pinned_tensor_rows(labels, seeds)
                else:
                    batch_inputs = features[input_nodes].to('cuda')
                    batch_labels = labels[seeds].to('cuda')
                torch.cuda.synchronize()
                epoch_feature_loading += time.time() - tic

                batch_pred, reg_loss = model(blocks, batch_inputs)
                is_labeled = batch_labels == batch_labels
                batch_labels = batch_labels[is_labeled].long()
                batch_pred = batch_pred[is_labeled]
                val_pred.append(batch_pred)
                val_labels.append(batch_labels)
                torch.cuda.synchronize()
                tic = time.time()

        acc = compute_acc(torch.cat(val_pred, 0),
                          torch.cat(val_labels, 0)).item()

        torch.cuda.synchronize()
        epoch_time.append(time.time() - start)
        sample_time_list.append(sample_time)
        mem_list.append((torch.cuda.max_memory_allocated() -
                        static_memory) / (1024 * 1024 * 1024))
        feature_loading_list.append(epoch_feature_loading)

        print("Epoch {:05d} | Val Acc {:.4f} | E2E Time {:.4f} s | Sampling Time {:.4f} s | Feature Loading Time {:.4f} s | GPU Mem Peak {:.4f} GB"
              .format(epoch, acc, epoch_time[-1], sample_time_list[-1], feature_loading_list[-1], mem_list[-1]))

    print('Average epoch end2end time:', np.mean(epoch_time[2:]))
    print('Average epoch sampling time:', np.mean(sample_time_list[2:]))
    print('Average epoch feature loading time:',
          np.mean(feature_loading_list[2:]))
    print('Average epoch gpu mem peak:', np.mean(mem_list[2:]))

    # print('Testing...')
    # acc = layerwise_infer(g, test_nid, model,
    #                       batch_size=4096, feat=features, label=labels)
    # print("Test Accuracy {:.4f}".format(acc.item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cuda', choices=['cuda', 'cpu'],
                        help="Training model on gpu or cpu")
    parser.add_argument('--use-uva', action=argparse.BooleanOptionalAction,
                        help="Wether to use UVA to sample graph and load feature")
    parser.add_argument("--dataset", default='reddit', choices=['reddit', 'products'],
                        help="which dataset to load for training")
    parser.add_argument("--batchsize", type=int, default=256,
                        help="batch size for training")
    parser.add_argument("--samples", default='2000,2000',
                        help="sample size for each layer")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="numbers of workers for sampling, must be 0 when gpu or uva is used")
    args = parser.parse_args()
    print(args)

    if args.dataset == 'reddit':
        dataset = load_reddit()
    elif args.dataset == 'products':
        dataset = load_ogbn_products()
    print(dataset[0])
    train(dataset, args)
