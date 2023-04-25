from gs import Graph, HeteroGraph, Matrix, HeteroMatrix
import gs
import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# g2 = dgl.heterograph({
#     ('user', 'follow', 'user'): ([0, 1, 2], [1, 2, 3]),
#     ('user', 'view', 'item'): ([0,1,2], [0,1,2]),
A1 = Graph(False)
indptr1 = torch.LongTensor([0, 0, 1, 2, 3]).to('cuda:0')
indices1 = torch.LongTensor([0, 1, 2]).to('cuda:0')
A1._CAPI_load_csc(indptr1, indices1)

A2 = Graph(False)
indptr2 = torch.LongTensor([0, 1, 2, 3]).to('cuda:0')
indices2 = torch.LongTensor([0, 1, 2]).to('cuda:0')
A2._CAPI_load_csc(indptr2, indices2)


node_types = ['user', 'item']
edge_types = [('user', 'follow', 'user'), ('user', 'view', 'item')]
graphs = [Matrix(A1), Matrix(A2)]
hg = HeteroGraph()
heteroM = HeteroMatrix(hg)

heteroM.load_from_homo(node_types, edge_types, graphs)

seeds = torch.LongTensor([2, 1, 0]).to('cuda:0')
print("random walk non_fused:")
nodes = heteroM.metapath_random_walk(seeds, ['view', 'follow', 'follow'])
print(nodes)

print("random walk fused:")
other_nodes = heteroM.metapath_random_walk_fused(
    seeds, ['view', 'follow', 'follow'])

print(other_nodes.reshape((-1, seeds.numel())))
