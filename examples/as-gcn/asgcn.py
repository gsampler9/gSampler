import gs
import torch
import torch.nn as nn
import torch.nn.functional as F
from gs.utils import load_graph
from typing import List


def asgcn_sampler(
    A: gs.Matrix,
    seeds: torch.Tensor,
    fanouts: List,
    features: torch.Tensor,
    W: torch.Tensor,
):
    output_nodes = seeds
    ret = []
    for K in fanouts:
        subA = A[:, seeds]
        subA.edata["w"] = subA.edata["w"] ** 2
        p = subA.sum("w", axis=1).sqrt()
        node_feats_u = features
        node_feats_v = features[seeds]
        h_u = node_feats_u @ W[:, 0]
        h_v = node_feats_v @ W[:, 1]
        h_v_sum = torch.sum(h_v)
        attention = torch.flatten((F.relu(h_u + h_v_sum) + 1) / K)
        g_u = torch.flatten(F.relu(h_u) + 1)
        q = F.normalize(p * attention * g_u, p=1.0, dim=0)

        sampleA = subA.collective_sampling(K, q, False)

        sampleA.edata["w"] = gs.ops.u_add_v(sampleA, h_u[sampleA.row_ndata["_ID"]], h_v)
        sampleA.edata["w"] = (F.relu(sampleA.edata["w"]) + 1) / sampleA.num_rows()
        sampleA = sampleA.div("w", q[sampleA.row_ndata["_ID"]], 1)

        seeds = sampleA.all_nodes()
        ret.append(sampleA.to_dgl_block())
    input_nodes = seeds
    return input_nodes, output_nodes, ret


if __name__ == "__main__":
    torch.manual_seed(1)
    dataset = load_graph.load_reddit()
    dgl_graph, features = dataset[0], dataset[1].cuda()
    csc_indptr, csc_indices, _ = dgl_graph.adj_sparse("csc")
    W = nn.init.xavier_normal_(torch.Tensor(features.shape[1], 2)).cuda()

    m = gs.Matrix()
    m.load_graph("CSC", [csc_indptr.cuda(), csc_indices.cuda()])
    m.edata["w"] = torch.ones(m.num_edges(), dtype=torch.float32).cuda()

    D_in = m.sum("w", axis=0)
    D_out = m.sum("w", axis=1)
    P = m.div("w", D_out.sqrt(), axis=1).div("w", D_in.sqrt(), axis=0)

    seeds = torch.randint(0, 10000, (512,)).cuda()

    # compile_func = gs.jit.compile(
    #    func=ladies_sampler, args=(m, seeds, [2000, 2000]))
    compile_func = asgcn_sampler
    for i in compile_func(m, seeds, [2000, 2000], features, W):
        print(i)
