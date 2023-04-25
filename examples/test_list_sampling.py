import gs
import torch

data = torch.tensor([0, 1, 2, 3, 4, 5]).long().cuda()

select, index = torch.ops.gs_ops.list_sampling(data, 20, True)

print(select)
print(index)

data = torch.tensor([0, 1, 2, 3, 4, 5]).long().cuda()
probs = torch.arange(6).float().cuda()

select, index = torch.ops.gs_ops.list_sampling_with_probs(
    data, probs, 3, False)

print(select)
print(index)