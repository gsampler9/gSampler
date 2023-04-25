from typing import List
import gs
import torch
import load_graph
import time
import numpy as np


def graphsaint(A: gs.Matrix, seeds_num, walk_length):
    seeds = torch.randint(
        0, 232965, (seeds_num,), device='cuda')
    paths = A.random_walk(seeds, walk_length)
    node_ids = paths.view(seeds_num*(walk_length+1))
    node_ids = node_ids[node_ids!=-1]
    out = torch.unique(node_ids, sorted=False)
    induced_subA = A[out, out]
    induced_subA.relabel()
    return induced_subA


dataset = load_graph.load_reddit()
dgl_graph = dataset[0]
m = gs.Matrix(gs.Graph(False))
m.load_dgl_graph(dgl_graph)
print("Check load successfully:", m._graph._CAPI_metadata(), '\n')


def bench(func, args):
    time_list = []
    for i in range(100):
        # print(i)
        torch.cuda.synchronize()
        begin = time.time()

        ret = func(*args)
        # print(ret._graph._CAPI_metadata())
        torch.cuda.synchronize()
        end = time.time()
        time_list.append(end - begin)
    print("fused graphsage sampling AVG:",
          np.mean(time_list[10:]) * 1000, " ms.")


bench(graphsaint, args=(
    m,
    2000,
    4,
))
