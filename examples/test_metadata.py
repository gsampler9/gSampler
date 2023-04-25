import gs
import torch

A = gs.Graph(False)
indptr = torch.LongTensor([0, 1, 1, 3, 4]).to('cuda:0')
indices = torch.LongTensor([3, 0, 1, 2]).to('cuda:0')
A._CAPI_load_csc(indptr, indices)

print(A._CAPI_metadata())