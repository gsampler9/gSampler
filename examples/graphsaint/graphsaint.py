import gs
import torch
from gs.utils import load_graph
from typing import List


def graphsaint_sampler(A: gs.matrix_api.Matrix, seeds: torch.Tensor, walk_length: int):
    paths = A._graph._CAPI_RandomWalk(seeds, walk_length)
    node_ids = paths.view(seeds.numel() * (walk_length + 1))
    node_ids = node_ids[node_ids != -1]
    out = torch.unique(node_ids, sorted=False)
    subA = A[out, out]
    return subA.to_dgl_block()


if __name__ == "__main__":
    torch.manual_seed(1)
    dataset = load_graph.load_reddit()
    dgl_graph = dataset[0]
    csc_indptr, csc_indices, _ = dgl_graph.adj_tensors("csc")

    m = gs.matrix_api.Matrix()
    m.load_graph("CSC", [csc_indptr.cuda(), csc_indices.cuda()])

    seeds = torch.randint(0, 10000, (2000,)).cuda()
    # compile_func = graphsaint_sampler
    compile_func = gs.jit.compile(func=graphsaint_sampler, args=(m, seeds, 4))
    print(compile_func.gm.code)

    subA = compile_func(m, seeds, 4)
    print(subA)
