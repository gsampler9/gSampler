import gs
import torch
from gs.utils import load_graph
from typing import List
import random

random.random()


def node2vec_sampler(A: gs.matrix_api.Matrix, seeds: torch.Tensor,
                     walk_length: int, p: float, q: float):
    paths = A.node2vec(seeds, walk_length, p, q)
    return paths


if __name__ == "__main__":
    torch.manual_seed(1)
    dataset = load_graph.load_reddit()
    dgl_graph = dataset[0]
    csc_indptr, csc_indices, _ = dgl_graph.adj_tensors("csc")

    m = gs.matrix_api.Matrix()
    m.load_graph("CSC", [csc_indptr.cuda(), csc_indices.cuda()])

    m._graph._CAPI_SortCSCIndices()
    seeds = torch.randint(0, 10000, (128, )).cuda()
    compile_func = gs.jit.compile(func=node2vec_sampler,
                                  args=(m, seeds, 80, 2.0, 0.5))
    paths = compile_func(m, seeds, 80, 2.0, 0.5)
    print(paths)
