from dataclasses import replace
import imp
from sys import meta_path
import torch
import load_graph
import dgl
from dgl.transforms.functional import to_block
from dgl.sampling import random_walk, pack_traces
import time
import numpy as np

device = torch.device('cuda:%d' % 0)

dataset = load_graph.load_reddit()
dgl_graph = dataset[0]
g = dgl_graph.long()
g = g.to("cuda")

#


def graphsaint_baseline(graph: dgl, num_roots, walk_length):
    sampled_roots = torch.randint(
        0, g.num_nodes(), (num_roots,), device='cuda')
    traces, types = random_walk(g, nodes=sampled_roots, length=walk_length)
    sampled_nodes = traces.view(num_roots*(walk_length+1))
    sampled_nodes = sampled_nodes[sampled_nodes!=-1]
    sg = graph.subgraph(sampled_nodes, relabel_nodes=True)
    return sg


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
    print("dgl graphSaint with %d seeds and %d metapath length AVG:" % (seeds_num, metalength),
          np.mean(time_list[10:]) * 1000, " ms.")


# seeds_set = [1000, 10000, 50000, 100000, 200000, 2000000, 10000000]
# metapath_len = [5, 10, 15, 20, 25, 30]
seeds_set = [2000]
metapath_len = [4]
for seed_num in seeds_set:
    for walk_length in metapath_len:
        bench(
            100,
            seed_num,
            walk_length,
            graphsaint_baseline, args=(
                g,
                seed_num,
                walk_length
            )
        )
print("seed_num,metapath_length,randomwalk_time")
for line in str_list:
    print(line)
