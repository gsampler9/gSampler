import gs
import torch
from gs.utils import load_graph
from typing import List


def ladies_sampler(A: gs.Matrix, seeds: torch.Tensor, fanouts: List):
    input_node = seeds
    ret = []
    for K in fanouts:
        subA = A[:, seeds]
        prob = subA.sum('w', axis=1) ** 2
        sampleA = subA.collective_sampling(K, prob, False)
        sampleA = sampleA.div('w', prob[sampleA.row_ndata['_ID']], axis=1)
        sampleA = sampleA.div('w', sampleA.sum('w', axis=0), axis=0)
        seeds = sampleA.all_nodes()
        ret.append(sampleA.to_dgl_block())
    output_node = seeds
    return input_node, output_node, ret


if __name__ == "__main__":
    torch.manual_seed(1)
    dataset = load_graph.load_reddit()
    dgl_graph = dataset[0]
    csc_indptr, csc_indices, _ = dgl_graph.adj_sparse('csc')

    m = gs.Matrix()
    m.load_graph('CSC', [csc_indptr.cuda(), csc_indices.cuda()])
    m.edata['w'] = torch.ones(m.num_edges(), dtype=torch.float32).cuda()
    
    D_in = m.sum('w', axis=0)
    D_out = m.sum('w', axis=1)
    P = m.div('w', D_out.sqrt(), axis=1).div('w', D_in.sqrt(), axis=0)

    seeds = torch.randint(0, 10000, (500,)).cuda()

    # compile_func = gs.jit.compile(
    #    func=ladies_sampler, args=(m, seeds, [2000, 2000]))
    compile_func = ladies_sampler
    for i in compile_func(m, seeds, [2000, 2000]):
        print(i)
