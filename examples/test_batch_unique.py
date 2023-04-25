import gs
import torch
import time
import numpy as np

tensor = torch.tensor([i for i in range(20000)]).long().cuda()
offset_ptr = torch.tensor([0, 20000]).long().cuda()
key_tensor = torch.tensor([0 for i in range(20000)]).long().cuda()

for i in torch.ops.gs_ops.BatchUniqueByKey(tensor, offset_ptr, key_tensor):
    print(i)

for i in torch.ops.gs_ops.BatchUnique(tensor, offset_ptr):
    print(i)

time_list = []
for i in range(50):
    begin = time.time()
    torch.ops.gs_ops.BatchUniqueByKey(tensor, offset_ptr, key_tensor)
    torch.cuda.synchronize()
    end = time.time()
    time_list.append(end - begin)
print(np.mean(time_list[5:]) * 1000)

time_list = []
for i in range(50):
    begin = time.time()
    torch.ops.gs_ops.BatchUnique(tensor, offset_ptr)
    torch.cuda.synchronize()
    end = time.time()
    time_list.append(end - begin)
print(np.mean(time_list[5:]) * 1000)

time_list = []
for i in range(50):
    begin = time.time()
    torch.unique(tensor)
    torch.cuda.synchronize()
    end = time.time()
    time_list.append(end - begin)
print(np.mean(time_list[5:]) * 1000)
