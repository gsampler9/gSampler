from dataclasses import replace
import imp
from sys import meta_path
import torch
import load_graph
import dgl
from dgl.transforms.functional import to_block
import time
import numpy as np

device = torch.device('cuda:%d' % 0)

dataset = load_graph.load_reddit()
dgl_graph = dataset[0]
g = dgl_graph.long()
g = g.to("cuda")

graph_data = {
    ('user', 'cite', 'user'): (g.srcnodes(), g.dstnodes())
}
hetero_g = dgl.heterograph(graph_data)
hetero_g.to("cuda")
print(hetero_g.canonical_etypes)
print(hetero_g)


def randomwalk_baseline(hetero_g, seeds, metapath):
    torch.cuda.nvtx.range_push("dgl random walk")
    ret = dgl.sampling.random_walk(hetero_g, nodes=seeds, metapath=metapath)
    torch.cuda.nvtx.range_pop()
    return ret


str_list = []


def bench(loop_num, seeds_num, metalength, func, args):
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
    print("dgl randomwalk with %d seeds and %d metapath length AVG:" % (seeds_num, metalength),
          np.mean(time_list[10:]) * 1000, " ms.")


seeds_set = [1000, 10000, 50000, 100000, 200000, 2000000, 10000000]
metapath_len = [5, 10, 15, 20, 25, 30]
# seeds_set = [200000]
# metapath_len = [30]
for seed_num in seeds_set:
    for metalenth in metapath_len:
        seeds = torch.randint(0, 232964, (seed_num,), device='cuda')
        metapath = ['cite']*metalenth
        bench(
            100,
            seed_num,
            metalenth,
            randomwalk_baseline, args=(
                hetero_g,
                seeds,
                metapath
            )
        )
print("seed_num,metapath_length,randomwalk_time")
for line in str_list:
    print(line)
