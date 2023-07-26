import gs
import torch
from gs.utils import load_graph
from typing import List


def graphsage_sampler(A: gs.matrix_api.Matrix, seeds: torch.Tensor, fanouts: List):
    input_node = seeds
    ret = []
    for K in fanouts:
        subA = A[:, seeds]
        sampleA = subA.individual_sampling(K, None, False)
        seeds = sampleA.all_nodes()
        ret.append(sampleA.to_dgl_block())
    output_node = seeds
    return input_node, output_node, ret


if __name__ == "__main__":
    torch.manual_seed(1)
    dataset = load_graph.load_reddit()
    dgl_graph = dataset[0]
    csc_indptr, csc_indices, _ = dgl_graph.adj_tensors("csc")

    m = gs.matrix_api.Matrix()
    m.load_graph("CSC", [csc_indptr.cuda(), csc_indices.cuda()])

    seeds = torch.randint(0, 10000, (1024,)).cuda()

    compile_func = gs.jit.compile(func=graphsage_sampler, args=(m, seeds, [25, 10]))
    print(compile_func.gm.graph)
    for i in compile_func(m, seeds, [25, 10]):
        print(i)
