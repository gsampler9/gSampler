import gs
from typing import List
import torch
import examples.load_graph as load_graph
import time
import numpy as np


def fastgcn(A: gs.Matrix, seeds: torch.Tensor, probs: torch.Tensor,
            fanouts: List):
    input_node = seeds
    ret = []
    for fanout in fanouts:
        subA = A[:, seeds]
        row_indices = subA.row_indices()
        selected, _ = torch.ops.gs_ops.list_sampling_with_probs(
            row_indices, probs[row_indices], fanout, False)
        subA = subA[selected, :]
        seeds = subA.all_indices()
        ret.append(subA)
    output_node = seeds
    return input_node, output_node, ret


dataset = load_graph.load_reddit()
dgl_graph = dataset[0]
m = gs.Matrix(gs.Graph(False))
m.load_dgl_graph(dgl_graph)
probs = dgl_graph.out_degrees().float().cuda()
print("Check load successfully:", m._graph._CAPI_metadata(), '\n')
print("probs: ", probs)
seeds = torch.arange(0, 1000).long().cuda()

compiled_func = gs.jit.compile(func=fastgcn,
                               args=(m, seeds, probs, [2000, 2000]))
#print(compiled_func.gm)


def bench(func, args):
    time_list = []
    for i in range(100):
        torch.cuda.synchronize()
        begin = time.time()
        ret = func(*args)
        torch.cuda.synchronize()
        end = time.time()
        time_list.append(end - begin)

    print("fastgcn sampling AVG:", np.mean(time_list[10:]) * 1000, " ms.")


bench(compiled_func, args=(m, seeds, probs, [2000, 2000]))
