import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.utils import gather_pinned_tensor_rows
from dgl.nn.pytorch import SAGEConv
import gs
from gs.utils import SeedGenerator, load_reddit, load_ogb, create_block_from_csc
import numpy as np
import time
import tqdm
import argparse


class DGLSAGEModel(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_size, hid_size, "mean"))
        for i in range(num_layers - 2):
            self.layers.append(SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)

    def forward(self, blocks, h):
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


def compute_acc(pred, label):
    return (pred.argmax(1) == label).float().mean()


def graphsage_sampler(A, fanouts, seeds, seeds_ptr):
    ret = []
    for k in fanouts:
        subA = A[:, seeds::seeds_ptr]
        sampleA = subA.individual_sampling(k, None, False)
        seeds, seeds_ptr = sampleA.all_nodes()
        ret.append(sampleA.to_dgl_block())
    return ret


def train(dataset, args):
    device = args.device
    use_uva = args.use_uva
    fanouts = [int(x.strip()) for x in args.samples.split(",")]

    g, features, labels, n_classes, splitted_idx = dataset
    train_nid, val_nid, _ = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    g = g.to(device)
    train_nid, val_nid = train_nid.to(device), val_nid.to(device)
    features, labels = features.to(device), labels.to(device)
    csc_indptr, csc_indices, edge_ids = g.adj_sparse("csc")
    if use_uva and device == "cpu":
        features, labels = features.pin_memory(), labels.pin_memory()
        csc_indptr = csc_indptr.pin_memory()
        csc_indices = csc_indices.pin_memory()
        train_nid, val_nid = train_nid.pin_memory(), val_nid.pin_memory()
    m = gs.Matrix()
    m.load_graph("CSC", [csc_indptr, csc_indices])
    bm = gs.BatchMatrix()
    bm.load_from_matrix(m, False)

    batch_size = 51200
    small_batch_size = args.batchsize
    num_batches = int((batch_size + small_batch_size - 1) / small_batch_size)
    orig_seeds_ptr = torch.arange(num_batches + 1, dtype=torch.int64, device="cuda") * small_batch_size
    print(batch_size, small_batch_size, fanouts)

    train_seedloader = SeedGenerator(train_nid, batch_size=batch_size, shuffle=True, drop_last=False)
    val_seedloader = SeedGenerator(val_nid, batch_size=batch_size, shuffle=True, drop_last=False)
    model = DGLSAGEModel(features.shape[1], 256, n_classes, len(fanouts)).to("cuda")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    compile_func = gs.jit.compile(func=graphsage_sampler, args=(bm, fanouts, train_seedloader.data[:batch_size], orig_seeds_ptr))

    torch.cuda.synchronize()
    static_memory = torch.cuda.memory_allocated()
    print("memory allocated before training:", static_memory / (1024 * 1024 * 1024), "GB")

    epoch_time = []
    cur_time = []
    acc_list = []
    start = time.time()
    for epoch in range(args.num_epoch):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        tic = time.time()
        model.train()
        num_batches = int((batch_size + small_batch_size - 1) / small_batch_size)
        for it, seeds in enumerate(tqdm.tqdm(train_seedloader)):
            seeds_ptr = orig_seeds_ptr
            if it == len(train_seedloader) - 1:
                num_batches = int((seeds.numel() + small_batch_size - 1) / small_batch_size)
                seeds_ptr = torch.arange(num_batches + 1, dtype=torch.int64, device="cuda") * small_batch_size
                seeds_ptr[-1] = seeds.numel()
            all_blocks = compile_func(bm, fanouts, seeds, seeds_ptr)

            for rank in range(num_batches):
                blocks = []
                for layer in range(len(fanouts)):
                    blocks.insert(0, all_blocks[layer][rank])
                input_nodes = blocks[0].srcdata["_ID"]
                output_nodes = seeds[seeds_ptr[rank] : seeds_ptr[rank + 1]].cuda()
                if use_uva:
                    batch_inputs = gather_pinned_tensor_rows(features, input_nodes)
                    batch_labels = gather_pinned_tensor_rows(labels, output_nodes)
                else:
                    batch_inputs = features[input_nodes].to("cuda")
                    batch_labels = labels[output_nodes].to("cuda")

                batch_pred = model(blocks, batch_inputs)
                is_labeled = batch_labels == batch_labels
                batch_labels = batch_labels[is_labeled].long()
                batch_pred = batch_pred[is_labeled]
                loss = F.cross_entropy(batch_pred, batch_labels)
                opt.zero_grad()
                loss.backward()
                opt.step()

        model.eval()
        val_pred = []
        val_labels = []
        with torch.no_grad():
            num_batches = int((batch_size + small_batch_size - 1) / small_batch_size)
            for it, seeds in enumerate(tqdm.tqdm(val_seedloader)):
                seeds_ptr = orig_seeds_ptr
                if it == len(val_seedloader) - 1:
                    num_batches = int((seeds.numel() + small_batch_size - 1) / small_batch_size)
                    seeds_ptr = torch.arange(num_batches + 1, dtype=torch.int64, device="cuda") * small_batch_size
                    seeds_ptr[-1] = seeds.numel()
                all_blocks = compile_func(bm, fanouts, seeds, seeds_ptr)

                for rank in range(num_batches):
                    blocks = []
                    for layer in range(len(fanouts)):
                        blocks.insert(0, all_blocks[layer][rank])
                    input_nodes = blocks[0].srcdata["_ID"]
                    output_nodes = seeds[seeds_ptr[rank] : seeds_ptr[rank + 1]].cuda()
                    if use_uva:
                        batch_inputs = gather_pinned_tensor_rows(features, input_nodes)
                        batch_labels = gather_pinned_tensor_rows(labels, output_nodes)
                    else:
                        batch_inputs = features[input_nodes].to("cuda")
                        batch_labels = labels[output_nodes].to("cuda")

                    batch_pred = model(blocks, batch_inputs)
                    is_labeled = batch_labels == batch_labels
                    batch_labels = batch_labels[is_labeled].long()
                    batch_pred = batch_pred[is_labeled]
                    val_pred.append(batch_pred)
                    val_labels.append(batch_labels)

        acc = compute_acc(torch.cat(val_pred, 0), torch.cat(val_labels, 0)).item()
        acc_list.append(acc)

        torch.cuda.synchronize()
        end = time.time()
        cur_time.append(end - start)
        epoch_time.append(end - tic)

        print("Epoch {:05d} | Val Acc {:.4f} | E2E Time {:.4f} s | Accumulated Time {:.4f} s".format(epoch, acc, epoch_time[-1], cur_time[-1]))

    torch.cuda.synchronize()
    total_time = time.time() - start

    print("Total Elapse Time:", total_time)
    print("Average Epoch Time:", np.mean(epoch_time[3:]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Training model on gpu or cpu",
    )
    parser.add_argument(
        "--use-uva",
        type=bool,
        default=False,
        help="Wether to use UVA to sample graph and load feature",
    )
    parser.add_argument(
        "--dataset",
        default="products",
        choices=["reddit", "products", "papers100m"],
        help="which dataset to load for training",
    )
    parser.add_argument("--batchsize", type=int, default=512, help="batch size for training")
    parser.add_argument("--samples", default="25,10", help="sample size for each layer")
    parser.add_argument("--num-epoch", type=int, default=100, help="numbers of epoch in training")
    args = parser.parse_args()
    print(args)

    if args.dataset == "reddit":
        dataset = load_reddit()
    elif args.dataset == "products":
        dataset = load_ogb("ogbn-products", "/home/ubuntu/dataset")
    elif args.dataset == "papers100m":
        dataset = load_ogb("ogbn-papers100M", "/home/ubuntu/dataset")
    print(dataset[0])
    train(dataset, args)
