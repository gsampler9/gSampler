import torch
import torch.nn.functional as F
from dgl.utils import gather_pinned_tensor_rows
import gs
from gs.utils import SeedGenerator, load_reddit, load_ogb, create_block_from_coo, ConvModel
import numpy as np
import time
import tqdm
import argparse

torch.manual_seed(1)


def compute_acc(pred, label):
    return (pred.argmax(1) == label).float().mean()


def sampler(A: gs.Graph, seeds, seeds_ptr, probs, fanouts, num_batches):
    seedsts, units, colts, rowts = [], [], [], []
    for layer, fanout in enumerate(fanouts):
        subA = A._CAPI_slicing(seeds, 0, gs._CSC, gs._CSC + gs._COO, False)

        indptr, indices, indices_ptr = subA.GetBatchCSC(seeds_ptr)
        encoded_indices = torch.ops.gs_ops.BatchEncode(indices, indices_ptr)
        # torch.cuda.nvtx.range_push('batch unique')
        neighbors = torch.unique(encoded_indices, sorted=True)
        neighbors_batch_mask = neighbors >> 48
        # torch.cuda.nvtx.range_pop()
        node_probs = probs[neighbors - (neighbors_batch_mask << 48)]
        # torch.cuda.nvtx.range_push('batch list sample')
        neighbors_ptr = torch.ops.gs_ops.GetBatchOffsets(
            neighbors_batch_mask, num_batches)
        selected, _, selected_ptr = torch.ops.gs_ops.batch_list_sampling_with_probs(
            neighbors, node_probs, fanout, False, neighbors_ptr)

        # slicing

        coo_row, coo_col = subA.GetBatchCOO()
        # torch.cuda.nvtx.range_push('batch coo slice')
        sub_coo_row, sub_coo_col = torch.ops.gs_ops.COORowSlicingGlobalId(
            encoded_indices, coo_col, selected)
        sub_coo_row_batch_mask = sub_coo_row >> 48
        sub_coo_ptr = torch.ops.gs_ops.GetBatchOffsets(
            sub_coo_row_batch_mask, num_batches)
        sub_coo_row = sub_coo_row - (sub_coo_row_batch_mask << 48)
        # torch.cuda.nvtx.range_pop()
        # relabel
        unique_tensor, unique_tensor_ptr, sub_coo_row, sub_coo_col, sub_coo_ptr = torch.ops.gs_ops.BatchCOORelabel(
            seeds, seeds_ptr, sub_coo_col, sub_coo_row, sub_coo_ptr)

        seedst = torch.ops.gs_ops.SplitByOffset(seeds, seeds_ptr)
        unit = torch.ops.gs_ops.SplitByOffset(unique_tensor,
                                              unique_tensor_ptr)
        colt = torch.ops.gs_ops.SplitByOffset(sub_coo_col, sub_coo_ptr)
        rowt = torch.ops.gs_ops.SplitByOffset(sub_coo_row, sub_coo_ptr)
        seedsts.append(seedst)
        units.append(unit)
        colts.append(colt)
        rowts.append(rowt)
        # torch.cuda.nvtx.range_pop()

        seeds, seeds_ptr = unique_tensor, unique_tensor_ptr
    return seedsts, units, colts, rowts


def train(dataset, args):
    device = args.device
    use_uva = args.use_uva
    fanouts = [int(x.strip()) for x in args.samples.split(',')]

    g, features, labels, n_classes, splitted_idx = dataset
    train_nid, val_nid, _ = splitted_idx['train'], splitted_idx[
        'valid'], splitted_idx['test']
    g = g.to(device)
    train_nid, val_nid = train_nid.to(device), val_nid.to(device)
    features, labels = features.to(device), labels.to(device)
    csc_indptr, csc_indices, edge_ids = g.adj_sparse('csc')
    probs = g.out_degrees().float().to(device)
    if use_uva and device == 'cpu':
        features, labels = features.pin_memory(), labels.pin_memory()
        csc_indptr = csc_indptr.pin_memory()
        csc_indices = csc_indices.pin_memory()
        train_nid, val_nid = train_nid.pin_memory(), val_nid.pin_memory()
        probs = probs.pin_memory()
    A = gs.Graph(False)
    A._CAPI_load_csc(csc_indptr, csc_indices)
    print("Check load successfully:", A._CAPI_metadata(), '\n')

    n_epoch = 5
    batch_size = 51200
    small_batch_size = args.batchsize
    num_batches = int((batch_size + small_batch_size - 1) / small_batch_size)
    orig_seeds_ptr = torch.arange(
        num_batches + 1, dtype=torch.int64, device='cuda') * small_batch_size
    print(n_epoch, batch_size, small_batch_size, fanouts)

    train_seedloader = SeedGenerator(train_nid,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     drop_last=False)
    val_seedloader = SeedGenerator(val_nid,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=False)
    model = ConvModel(features.shape[1], 64, n_classes,
                      len(fanouts)).to('cuda')
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

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
        torch.cuda.synchronize()
        tic = time.time()
        num_batches = int(
            (batch_size + small_batch_size - 1) / small_batch_size)
        for it, seeds in enumerate(tqdm.tqdm(train_seedloader)):
            seeds = seeds.to('cuda')
            seeds_ptr = orig_seeds_ptr
            if it == len(train_seedloader) - 1:
                num_batches = int(
                    (seeds.numel() + small_batch_size - 1) / small_batch_size)
                seeds_ptr = torch.arange(num_batches + 1,
                                         dtype=torch.int64,
                                         device='cuda') * small_batch_size
                seeds_ptr[-1] = seeds.numel()
            seeds, units, colts, rowts = sampler(
                A, seeds, seeds_ptr, probs, fanouts, num_batches)
            torch.cuda.synchronize()
            sample_time += time.time() - tic

            for rank in range(num_batches):
                batch_seeds = seeds[0][rank]
                blocks = []
                for layer in range(len(fanouts)):
                    layer_seeds, unique, col, row = seeds[layer][rank], units[
                        layer][rank], colts[layer][rank], rowts[layer][rank]
                    block = create_block_from_coo(row,
                                                  col,
                                                  num_src=unique.numel(),
                                                  num_dst=layer_seeds.numel())
                    block.srcdata['_ID'] = unique
                    blocks.insert(0, block)
                tic = time.time()
                blocks = [block.to('cuda') for block in blocks]
                if use_uva:
                    batch_inputs = gather_pinned_tensor_rows(
                        features, blocks[0].srcdata['_ID'])
                    batch_labels = gather_pinned_tensor_rows(
                        labels, batch_seeds)
                else:
                    batch_inputs = features[blocks[0].srcdata['_ID']].to(
                        'cuda')
                    batch_labels = labels[batch_seeds].to('cuda')
                torch.cuda.synchronize()
                epoch_feature_loading += time.time() - tic

                batch_pred = model(blocks, batch_inputs)
                is_labeled = batch_labels == batch_labels
                batch_labels = batch_labels[is_labeled].long()
                batch_pred = batch_pred[is_labeled]
                loss = F.cross_entropy(batch_pred, batch_labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            tic = time.time()

        model.eval()
        val_pred = []
        val_labels = []
        with torch.no_grad():
            torch.cuda.synchronize()
            tic = time.time()
            num_batches = int(
                (batch_size + small_batch_size - 1) / small_batch_size)
            for it, seeds in enumerate(tqdm.tqdm(val_seedloader)):
                seeds = seeds.to('cuda')
                seeds_ptr = orig_seeds_ptr
                if it == len(val_seedloader) - 1:
                    num_batches = int((seeds.numel() + small_batch_size - 1) /
                                      small_batch_size)
                    seeds_ptr = torch.arange(num_batches + 1,
                                             dtype=torch.int64,
                                             device='cuda') * small_batch_size
                    seeds_ptr[-1] = seeds.numel()
                seeds, units, colts, rowts = sampler(
                    A, seeds, seeds_ptr, probs, fanouts, num_batches)
                torch.cuda.synchronize()
                sample_time += time.time() - tic

                for rank in range(num_batches):
                    batch_seeds = seeds[0][rank]
                    blocks = []
                    for layer in range(len(fanouts)):
                        layer_seeds, unique, col, row = seeds[layer][rank], units[
                            layer][rank], colts[layer][rank], rowts[layer][rank]
                        block = create_block_from_coo(row,
                                                      col,
                                                      num_src=unique.numel(),
                                                      num_dst=layer_seeds.numel())
                        block.srcdata['_ID'] = unique
                        blocks.insert(0, block)
                    tic = time.time()
                    blocks = [block.to('cuda') for block in blocks]
                    if use_uva:
                        batch_inputs = gather_pinned_tensor_rows(
                            features, blocks[0].srcdata['_ID'])
                        batch_labels = gather_pinned_tensor_rows(
                            labels, batch_seeds)
                    else:
                        batch_inputs = features[blocks[0].srcdata['_ID']].to(
                            'cuda')
                        batch_labels = labels[batch_seeds].to('cuda')
                    torch.cuda.synchronize()
                    epoch_feature_loading += time.time() - tic

                    batch_pred = model(blocks, batch_inputs)
                    is_labeled = batch_labels == batch_labels
                    batch_labels = batch_labels[is_labeled].long()
                    batch_pred = batch_pred[is_labeled]
                    val_pred.append(batch_pred)
                    val_labels.append(batch_labels)
                torch.cuda.synchronize()
                tic = time.time()

        acc = compute_acc(torch.cat(val_pred, 0), torch.cat(val_labels,
                                                            0)).item()

        torch.cuda.synchronize()
        epoch_time.append(time.time() - start)
        sample_time_list.append(sample_time)
        mem_list.append((torch.cuda.max_memory_allocated() - static_memory) /
                        (1024 * 1024 * 1024))
        feature_loading_list.append(epoch_feature_loading)

        print(
            "Epoch {:05d} | Val Acc {:.4f} | E2E Time {:.4f} s | Sampling Time {:.4f} s | Feature Loading Time {:.4f} s | GPU Mem Peak {:.4f} GB"
            .format(epoch, acc, epoch_time[-1], sample_time_list[-1],
                    feature_loading_list[-1], mem_list[-1]))

    print('Average epoch end2end time:', np.mean(epoch_time[2:]))
    print('Average epoch sampling time:', np.mean(sample_time_list[2:]))
    print('Average epoch feature loading time:',
          np.mean(feature_loading_list[2:]))
    print('Average epoch gpu mem peak:', np.mean(mem_list[2:]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",
                        default='cuda',
                        choices=['cuda', 'cpu'],
                        help="Training model on gpu or cpu")
    parser.add_argument(
        '--use-uva',
        action=argparse.BooleanOptionalAction,
        help="Wether to use UVA to sample graph and load feature")
    parser.add_argument("--dataset",
                        default='products',
                        choices=['reddit', 'products', 'papers100m'],
                        help="which dataset to load for training")
    parser.add_argument("--batchsize",
                        type=int,
                        default=1024,
                        help="batch size for training")
    parser.add_argument("--samples",
                        default='2000,2000',
                        help="sample size for each layer")
    args = parser.parse_args()
    print(args)

    if args.dataset == 'reddit':
        dataset = load_reddit()
    elif args.dataset == 'products':
        dataset = load_ogb(
            'ogbn-products', '/home/ubuntu/gs-experiments/datasets')
    elif args.dataset == 'papers100m':
        dataset = load_ogb('ogbn-papers100M',
                           '/home/ubuntu/gs-experiments/datasets')
    print(dataset[0])
    train(dataset, args)
