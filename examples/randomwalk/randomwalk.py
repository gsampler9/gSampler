import gs
import torch
from gs.utils import load_graph
from typing import List


def randomwalk_sampler(A: gs.matrix_api.Matrix, seeds: torch.Tensor, walk_length:int):
    paths = A.random_walk(seeds, walk_length)
    return paths


if __name__ == "__main__":
    torch.manual_seed(1)
    dataset = load_graph.load_reddit()
    dgl_graph = dataset[0]
    csc_indptr, csc_indices, _ = dgl_graph.adj_sparse("csc")

    m = gs.matrix_api.Matrix()
    m.load_graph("CSC", [csc_indptr.cuda(), csc_indices.cuda()])

    seeds = torch.randint(0, 10000, (128,)).cuda()
    compile_func = randomwalk_sampler
    paths = compile_func(m, seeds, 80)
    print(paths)

