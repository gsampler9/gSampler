import gs
import torch

A = gs.Graph(False)
indptr = torch.tensor([0, 10, 11, 20, 22, 22]).long().cuda()
indices = torch.arange(22).long().cuda()
probs = torch.ones_like(indices).cuda().float()
probs[0] = 100

A._CAPI_load_csc(indptr, indices)

subA = A._CAPI_columnwise_sampling_with_probs(probs, 50, True)

for i in subA._CAPI_metadata():
    print(i)