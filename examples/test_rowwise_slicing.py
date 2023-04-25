import gs
import torch

A = gs.Graph(False)
indptr = torch.arange(21).long().cuda() * 5
indices = torch.arange(0, 100).long().cuda()
A._CAPI_load_csc(indptr, indices)

print(A._CAPI_metadata())

row_ids = torch.arange(0, 100, 2).long().cuda()
subA = A._CAPI_rowwise_slicing(row_ids)
print(subA._CAPI_metadata())