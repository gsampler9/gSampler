from dataclasses import replace
import imp
from sys import meta_path
import torch
import load_graph
import dgl
from dgl.transforms.functional import to_block
import time
import numpy as np
from gs import Graph, HeteroGraph, Matrix, HeteroMatrix
import gs

device = torch.device('cuda:%d' % 0)
dataset = load_graph.load_reddit()
dgl_graph = dataset[0]
g = dgl_graph.long()
reverse_g = dgl.reverse(g)
reverse_g = reverse_g.formats(['csr'])
csc = reverse_g.adj(scipy_fmt='csr')
csc_indptr = torch.tensor(csc.indptr).long().cuda()
csc_indices = torch.tensor(csc.indices).long().cuda()
node_types = ['user']
edge_types = [('user', 'cite', 'user')]
A1 = Graph(False)
A1._CAPI_load_csc(csc_indptr, csc_indices)
graphs = [Matrix(A1)]

hg = HeteroGraph()
heteroM = HeteroMatrix(hg)
heteroM.load_from_homo(node_types, edge_types, graphs)


def randomwalk_baseline(heteroM: HeteroMatrix, seeds, metapath):
    torch.cuda.nvtx.range_push("random walk python")
    ret = heteroM.metapath_random_walk(seeds, metapath)
    torch.cuda.nvtx.range_pop()
    return ret


str_list = []


def bench(loop_num,  seed_num,
          metalength, func, args, ):
    time_list = []
    for i in range(loop_num):
        torch.cuda.synchronize()
        begin = time.time()

        ret = func(*args)
        #print("ret:", ret)
        torch.cuda.synchronize()
        end = time.time()

        time_list.append(end - begin)
    str_list.append("%d,%d,%.3f" %
                    (seed_num, metalength, np.mean(time_list[10:]) * 1000))
    print("Non-fused randomwalk with %d seeds and %d metapath length AVG:" % (seed_num, metalength),
          np.mean(time_list[10:]) * 1000, " ms.")


# seeds_set = [1000, 10000, 50000, 100000, 200000, ]
# metapath_len = [5, 10, 15, 20, 25, 30]
seeds_set = [1000, 10000, 50000, 100000, 200000, 2000000, 10000000]
metapath_len = [5, 10, 15, 20, 25, 30]
seeds_set = [10000000]
metapath_len = [30]
for seed_num in seeds_set:
    for metalenth in metapath_len:
        seeds = torch.randint(0, 232964, (seed_num,), device='cuda')
        metapath = ['cite']*metalenth
        bench(
            100,
            seed_num,
            metalenth,
            randomwalk_baseline, args=(
                heteroM,
                seeds,
                metapath
            )
        )
for line in str_list:
    print(line)
