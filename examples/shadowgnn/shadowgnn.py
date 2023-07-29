import gs
import torch
from gs.utils import load_graph
from typing import List


def shadowgnn_sampler(A: gs.matrix_api.Matrix, seeds: torch.Tensor,
                      fanouts: List):
    output_nodes = seeds
    for K in fanouts:
        subA = A[:, seeds]
        sampleA = subA.individual_sampling(K, None, False)
        seeds = sampleA.all_nodes()
    input_nodes = seeds
    subA = A[seeds, seeds]
    return input_nodes, output_nodes, subA.to_dgl_block()


def batch_shadowgnn_sampler(A, seeds, seeds_ptr, fanouts):
    for k in fanouts:
        subA = A[:, seeds::seeds_ptr]
        sampleA = subA.individual_sampling(k, None, False)
        seeds, seeds_ptr = sampleA.all_nodes()
    subA = A[seeds::seeds_ptr, seeds::seeds_ptr]
    return subA.to_dgl_block()


if __name__ == "__main__":
    torch.manual_seed(1)
    dataset = load_graph.load_reddit()
    dgl_graph = dataset[0]
    csc_indptr, csc_indices, _ = dgl_graph.adj_tensors("csc")

    m = gs.matrix_api.Matrix()
    m.load_graph("CSC", [csc_indptr.cuda(), csc_indices.cuda()])

    seeds = torch.randint(0, 10000, (1024, )).cuda()
    seeds_ptr = torch.tensor([0, 512, 1024]).long().cuda()
    compile_func = gs.jit.compile(func=shadowgnn_sampler,
                                  args=(m, seeds, [25, 10]))
    print(compile_func.gm.code)
    for i in compile_func(m, seeds, [25, 10]):
        print(i)

    bm = gs.matrix_api.BatchMatrix()
    bm.load_from_matrix(m, False)

    bm_compile_func = gs.jit.compile(func=batch_shadowgnn_sampler,
                                     args=(bm, seeds, seeds_ptr, [25, 10]))

    print(bm_compile_func.gm.graph)

    for i in bm_compile_func(bm, seeds, seeds_ptr, [25, 10]):
        print(i)
