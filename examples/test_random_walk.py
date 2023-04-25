from gs import Graph, HeteroGraph, Matrix, HeteroMatrix
import gs
import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

A1 = Graph(False)
indptr1 = torch.LongTensor([0, 0, 1, 2, 3]).to('cuda:0')
indices1 = torch.LongTensor([0, 1, 2]).to('cuda:0')
A1.load_csc(indptr1, indices1)
m = Matrix(A1)

seeds = torch.LongTensor([3, 2]).to('cuda:0')
print("random walk fused:")
nodes = m.random_walk(seeds, 2)
print(nodes.reshape((-1, seeds.numel())))
