import gs
import torch
import time
import numpy as np

data1 = torch.tensor([0, 2, 3, 20, 21, 22, 21, 22] +
                     [25, 26, 27, 22, 22]).long().cuda()
offset_ptr1 = torch.tensor([0, 8, 13]).long().cuda()
key_tensor1 = torch.tensor([0 for _ in range(8)] +
                           [1 for _ in range(5)]).long().cuda()

data = torch.tensor([20, 21, 22, 21, 22] + [25, 26, 27, 22, 22]).long().cuda()
offset_ptr = torch.tensor([0, 5, 10]).long().cuda()
key_tensor = torch.tensor([0 for _ in range(5)] +
                          [1 for _ in range(5)]).long().cuda()

for i in torch.ops.gs_ops.BatchRelabelByKey(data1, offset_ptr1, key_tensor1,
                                            data, offset_ptr, key_tensor):
    print(i)

for i in torch.ops.gs_ops.BatchCSRRelabelByKey(data1, offset_ptr1, key_tensor1,
                                               data, offset_ptr, key_tensor):
    print(i)

for i in torch.ops.gs_ops.BatchCSRRelabel(data1, offset_ptr1, data,
                                          offset_ptr):
    print(i)

for i in torch.ops.gs_ops.BatchCOORelabelByKey(data1, offset_ptr1, key_tensor1,
                                               data, data, offset_ptr,
                                               key_tensor):
    print(i)

for i in torch.ops.gs_ops.BatchCOORelabel(data1, offset_ptr1, data, data,
                                          offset_ptr):
    print(i)
