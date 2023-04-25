import gs
import torch
import time
import numpy as np

tensor1 = torch.tensor([i for i in range(256 * 256)]).long().cuda()
tensor2 = torch.tensor([i for i in range(256 * 256)]).long().cuda()
offset_ptr1 = torch.tensor([i * 256 for i in range(257)]).long().cuda()
offset_ptr2 = torch.tensor([i * 256 for i in range(257)]).long().cuda()

print(tensor1)
print(offset_ptr1)

for i in torch.ops.gs_ops.BatchConcat([tensor1, tensor2],
                                      [offset_ptr1, offset_ptr2]):
    torch.cuda.synchronize()
    print(i)

time_list = []
for i in range(100):
    begin = time.time()
    torch.ops.gs_ops.BatchConcat([tensor1, tensor2],
                                 [offset_ptr1, offset_ptr2])
    torch.cuda.synchronize()
    end = time.time()

    time_list.append(end - begin)

print(np.mean(time_list[10:]) * 1000)

time_list = []
for i in range(100):
    begin = time.time()
    cat = torch.cat([tensor1, tensor2])
    out = offset_ptr1 + offset_ptr2
    torch.cuda.synchronize()
    end = time.time()

    time_list.append(end - begin)

print(np.mean(time_list[10:]) * 1000)
