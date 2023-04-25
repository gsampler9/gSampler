from typing import List
import gs
import torch
import examples.load_graph as load_graph
import time
import numpy as np


def graphsage(A: gs.Matrix, seeds: torch.Tensor, fanouts: List):
    input_node = seeds
    ret = []
    for fanout in fanouts:
        subA = A[:, seeds]
        subA = subA.columnwise_sampling(fanout, True)
        seeds = subA.all_indices()
        ret.append(subA)  # [todo] maybe bug before subA.row_indices
    output_node = seeds
    return input_node, output_node, ret


dataset = load_graph.load_reddit()
dgl_graph = dataset[0]
m = gs.Matrix(gs.Graph(False))
m.load_dgl_graph(dgl_graph)
print("Check load successfully:", m._graph._CAPI_metadata(), '\n')
seeds = torch.arange(0, 5000).long().cuda()

compiled_func = gs.jit.compile(func=graphsage, args=(m, seeds, [25, 15]))


def bench(func, args):
    time_list = []
    for i in range(100):
        #   print(i)
        torch.cuda.synchronize()
        begin = time.time()

        ret = func(*args)

        torch.cuda.synchronize()
        end = time.time()

        time_list.append(end - begin)

    print("nonfused graphsage sampling AVG:",
          np.mean(time_list[10:]) * 1000, " ms.")


bench(compiled_func, args=(
    m,
    seeds,
    [25, 15],
))
