import gs
import torch
import time
import numpy as np

data = torch.tensor([i for i in range(100)]).long().cuda()

tensor1 = torch.arange(10).long().cuda()
tensor2 = torch.arange(10).long().cuda()
tensor3 = torch.arange(10).long().cuda()
offset_ptr1 = torch.tensor([0, 3, 10]).long().cuda()
offset_ptr2 = torch.tensor([0, 6, 10]).long().cuda()
offset_ptr3 = torch.tensor([0, 9, 10]).long().cuda()

data, key, ptr = torch.ops.gs_ops.BatchConcat(
    [tensor1, tensor2, tensor3], [offset_ptr1, offset_ptr2, offset_ptr3])

print("Concat")
print(data)
print(key)
print(ptr)

out1 = torch.empty_like(tensor1)
out2 = torch.empty_like(tensor2)
out3 = torch.empty_like(tensor3)

torch.ops.gs_ops.BatchSplit(data, ptr, key, [out1, out2, out3],
                            [offset_ptr1, offset_ptr2, offset_ptr3])

print(out1)
print(out2)
print(out3)