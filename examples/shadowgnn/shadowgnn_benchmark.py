from dataclasses import replace
import imp
import gs
from sys import meta_path
import torch
import load_graph
import dgl
from dgl.transforms.functional import to_block
from dgl.sampling import random_walk, pack_traces
import time
import numpy as np
import argparse


def main(args):
    dataset = load_graph.load_reddit()
    dgl_graph = dataset[0]
    dgl_g = dgl_graph.long()
    dgl_g = dgl_g.to("cuda")
    matrix = gs.Matrix(gs.Graph(False))
    matrix.load_dgl_graph(dgl_graph)
    seed_tensor = torch.randint(
        0,  232965, (args.loops, args.seed_num), device='cuda', dtype=torch.int64)

    def shadowgnn_dgl(graph: dgl.DGLGraph, fanouts, seeds):
        output_nodes = seeds
        for fanout in reversed(fanouts):
            torch.cuda.nvtx.range_push("shadowgnn dgl sample neighbors")
            frontier = graph.sample_neighbors(seeds, fanout)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("shadowgnn dgl to block")
            block = dgl.transforms.to_block(frontier, seeds)
            torch.cuda.nvtx.range_pop()
            seeds = block.srcdata[dgl.NID]
        torch.cuda.nvtx.range_push("shadowgnn dgl induce subgraph")
        subg = graph.subgraph(seeds, relabel_nodes=True)
        torch.cuda.nvtx.range_pop()
        return seeds, output_nodes, subg

    def shadowgnn_nonfused(A: gs.Matrix, fanouts, seeds):
        output_nodes = seeds
        for fanout in reversed(fanouts):
            torch.cuda.nvtx.range_push(
                "shadowgnn columnwise slicing and sampling")
            subA = A[:, seeds]
            subA = subA.columnwise_sampling(fanout, False)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("shadowgnn all_indices")
            seeds = subA.all_indices()
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("shadowgnn induce subgraph")
        retA = A[seeds, seeds]
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("shadowgnn subgraph relabel")
        retA.relabel()
        torch.cuda.nvtx.range_pop()
        return seeds, output_nodes, retA

    def shadowgnn_fused(A: gs.Matrix, fanouts, seeds):
        output_nodes = seeds
        for fanout in reversed(fanouts):
            torch.cuda.nvtx.range_push(
                "shadowgnn columnwise slicing and sampling")
            subA = A.fused_columnwise_slicing_sampling(seeds, fanout, False)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("shadowgnn all_indices")
            seeds = subA.all_indices()
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("shadowgnn induce subgraph")
        retA = A[seeds, seeds]
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("shadowgnn subgraph relabel")
        retA.relabel()
        torch.cuda.nvtx.range_pop()
        return seeds, output_nodes, retA

    def bench(loop_num, func, args):
        time_list = []
        seed_tensor = args[2]
        for i in range(loop_num):
            torch.cuda.synchronize()
            begin = time.time()
            seed_nodes, output_nodes, subg = func(
                args[0], args[1], seed_tensor[i, :])
            #print("ret nodes:", seed_nodes.numel())
            torch.cuda.synchronize()
            end = time.time()
            time_list.append(end - begin)
        print(func.__name__, " AVG:", np.mean(time_list[10:]) * 1000, " ms.")
    fanouts = [5, 15, 25]
    if args.sample_alg == "dgl":
        bench(args.loops,
              shadowgnn_dgl, args=(
                  dgl_g,
                  fanouts, seed_tensor))
    elif args.sample_alg == "nonfused":
        bench(args.loops,
              shadowgnn_nonfused, args=(
                  matrix,
                  fanouts, seed_tensor))
    elif args.sample_alg == "fused":
        bench(args.loops,
              shadowgnn_fused, args=(
                  matrix,
                  fanouts, seed_tensor))
    else:
        bench(args.loops,
              shadowgnn_dgl, args=(
                  dgl_g,
                  fanouts, seed_tensor))
        bench(args.loops,
              shadowgnn_nonfused, args=(
                  matrix,
                  fanouts, seed_tensor))
        bench(args.loops,
              shadowgnn_fused, args=(
                  matrix,
                  fanouts, seed_tensor))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAINT')
    # sampler params
    parser.add_argument("--sample_alg", type=str, default="all", choices=['dgl', 'nonfused', 'fused', 'all'],
                        help="Type of sample algorithm")
    # training params
    parser.add_argument("--loops", type=int, default=100,
                        help="Number of test loops")
    parser.add_argument("--seed_num", type=int, default=5,
                        help="Number of seed num")
    args = parser.parse_args()
    main(args)
