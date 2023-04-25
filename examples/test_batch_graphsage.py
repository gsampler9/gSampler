import numpy as np
import gs
import torch
from gs.utils import create_block_from_csc
from gs.utils import SeedGenerator, load_reddit
import numpy as np
import time

torch.manual_seed(1)

g, features, labels, n_classes, splitted_idx = load_reddit()

train_nid = splitted_idx['train'].long().cuda()
indptr, indices, _ = g.adj_sparse('csc')

batch_size = 256 * 256
small_batch_size = 256
num_batchs = int(batch_size / small_batch_size)

fanout = 15

seeds_ptr = torch.arange(num_batchs + 1, dtype=torch.int64,
                         device='cuda') * small_batch_size

A = gs.Graph(False)
A._CAPI_load_csc(indptr.long().cuda(), indices.long().cuda())

seeds = train_nid[:batch_size]

# sampling
subA = A._CAPI_fused_columnwise_slicing_sampling(seeds, fanout, False)

# batch relabel
indptr, indices, indices_ptr = subA.GetBatchCSC(seeds_ptr)
data, data_key, data_ptr = torch.ops.gs_ops.BatchConcat(
    [seeds, indices], [seeds_ptr, indices_ptr])
unique_tensor, unique_tensor_ptr, relabel_data, relabel_data_ptr = torch.ops.gs_ops.BatchRelabelByKey(
    data, data_ptr, data_key, data, data_ptr, data_key)
torch.ops.gs_ops.BatchSplit(relabel_data, relabel_data_ptr, data_key,
                            [seeds, indices], [seeds_ptr, indices_ptr])

# to dgl block
unit = torch.ops.gs_ops.SplitByOffset(unique_tensor, unique_tensor_ptr)
ptrt = torch.ops.gs_ops.IndptrSplitByOffset(indptr, seeds_ptr)
indt = torch.ops.gs_ops.SplitByOffset(indices, indices_ptr)

for unique, indptr, indices in zip(unit, ptrt, indt):
    block = create_block_from_csc(indptr,
                                  indices,
                                  torch.tensor([]),
                                  num_src=unique.numel(),
                                  num_dst=indptr.numel() - 1)
    block.srcdata['_ID'] = unique
    print(block)
