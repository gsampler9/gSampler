import gs
import torch
import torch.nn as nn
import torch.nn.functional as F
from gs.utils import load_graph
from typing import List
from gs.format import _CSR, _CSC, _COO


def pass_sampler(
    A: gs.Matrix,
    seeds: torch.Tensor,
    fanouts: List,
    features: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    W3: torch.Tensor,
):
    ret = []
    output_nodes = seeds
    for K in fanouts:
        subA = A[:, seeds]
        u_feats = features
        v_feats = features[seeds]
        att1 = gs.ops.u_mul_v(subA, u_feats @ W1, v_feats @ W1, _COO)
        att2 = gs.ops.u_mul_v(subA, u_feats @ W2, v_feats @ W2, _COO)
        att1 = torch.sum(att1, dim=1)
        att2 = torch.sum(att2, dim=1)
        att3 = subA.div("w", subA.sum("w", axis=0), axis=0).edata["w"]
        att = torch.stack([att1, att2, att3], dim=1)
        att = F.relu(att @ F.softmax(W3, dim=0))
        att = att + 10e-10 * torch.ones_like(att)
        subA.edata["w"] = att

        sampleA = subA.individual_sampling(K, probs=att, replace=True)
        seeds = sampleA.all_nodes()
        ret.append(sampleA.to_dgl_block(prefetch_edata={"w"}))
    input_nodes = seeds
    return input_nodes, output_nodes, ret


if __name__ == "__main__":
    torch.manual_seed(1)
    dataset = load_graph.load_reddit()
    dgl_graph, features = dataset[0], dataset[1].cuda()
    csc_indptr, csc_indices, _ = dgl_graph.adj_tensors("csc")
    print("feature shape:", features.shape)
    in_size = features.shape[1]
    hid_size = 64
    W1 = nn.init.xavier_uniform_(torch.Tensor(size=(in_size, hid_size))).cuda()
    print("w1:", W1.dtype)
    W2 = nn.init.xavier_uniform_(torch.Tensor(size=(in_size, hid_size))).cuda()
    W3 = torch.FloatTensor([[10e-3], [10e-3], [10e-1]]).cuda()
    print("W shape:", W1.shape, W2.shape)

    m = gs.Matrix()
    m.load_graph("CSC", [csc_indptr.cuda(), csc_indices.cuda()])
    m.edata["w"] = torch.ones(m.num_edges(), dtype=torch.float32).cuda()

    seeds = torch.randint(0, 10000, (64,)).cuda()

    compile_func = gs.jit.compile(
        func=pass_sampler, args=(m, seeds, [25, 10], features, W1, W2, W3)
    )
    print(compile_func.gm.graph)
    for i in pass_sampler(m, seeds, [25, 10], features, W1, W2, W3):
        print(i)
