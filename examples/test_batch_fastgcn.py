import numpy as np
import gs
import torch
from gs.utils import create_block_from_coo
from gs.utils import SeedGenerator, load_reddit
import numpy as np
import time

torch.manual_seed(1)

g, features, labels, n_classes, splitted_idx = load_reddit()
probs = g.out_degrees().float().cuda()
train_nid = splitted_idx['train'].long().cuda()
indptr, indices, _ = g.adj_sparse('csc')

batch_size = 256 * 256
small_batch_size = 256
num_batchs = int(batch_size / small_batch_size)

fanout = 1000

seeds_ptr = torch.arange(num_batchs + 1, dtype=torch.int64,
                         device='cuda') * small_batch_size

A = gs.Graph(False)
A._CAPI_load_csc(indptr.long().cuda(), indices.long().cuda())

seeds = train_nid[:batch_size]

# slicing
subA = A._CAPI_slicing(seeds, 0, gs._CSC, gs._CSC + gs._COO, False)

# list sampling
indptr, indices, indices_ptr = subA.GetBatchCSC(seeds_ptr)
neighbors, neighbors_ptr, neighbors_key = torch.ops.gs_ops.BatchUnique(
    indices, indices_ptr)
node_probs = probs[neighbors]
selected, _, selected_ptr = torch.ops.gs_ops.batch_list_sampling_with_probs(
    neighbors, node_probs, fanout, False, neighbors_ptr)

# slicing
coo_row, coo_col = subA.GetBatchCOO()
sub_coo_row, sub_coo_col, sub_coo_ptr = torch.ops.gs_ops.BatchCOOSlicing(
    1, coo_row, coo_col, indices_ptr, selected, selected_ptr)

# relabel
mapping_data, mapping_data_key, mapping_data_ptr = torch.ops.gs_ops.BatchConcat(
    [seeds, sub_coo_row], [seeds_ptr, sub_coo_ptr])
data, data_key, data_ptr = torch.ops.gs_ops.BatchConcat(
    [sub_coo_col, sub_coo_row], [sub_coo_ptr, sub_coo_ptr])
unique_tensor, unique_tensor_ptr, relabel_data, relabel_data_ptr = torch.ops.gs_ops.BatchRelabelByKey(
    mapping_data, mapping_data_ptr, mapping_data_key, data, data_ptr, data_key)
torch.ops.gs_ops.BatchSplit(relabel_data, relabel_data_ptr, data_key,
                            [sub_coo_col, sub_coo_row],
                            [sub_coo_ptr, sub_coo_ptr])

seedst = torch.ops.gs_ops.SplitByOffset(seeds, seeds_ptr)
unit = torch.ops.gs_ops.SplitByOffset(unique_tensor, unique_tensor_ptr)
colt = torch.ops.gs_ops.SplitByOffset(sub_coo_col, sub_coo_ptr)
rowt = torch.ops.gs_ops.SplitByOffset(sub_coo_row, sub_coo_ptr)
# torch.cuda.nvtx.range_pop()
for s, unique, col, row in zip(seedst, unit, colt, rowt):
    block = create_block_from_coo(row,
                                  col,
                                  num_src=unique.numel(),
                                  num_dst=s.numel())
    block.srcdata['_ID'] = unique
    print(block)
