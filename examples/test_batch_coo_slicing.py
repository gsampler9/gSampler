import gs
import torch
import time
import numpy as np

coo_row = torch.tensor([20, 21, 22, 21, 22] +
                       [25, 26, 27, 22, 22]).long().cuda()
coo_col = torch.tensor([2, 1, 2, 2, 1] + [3, 3, 4, 34, 5]).long().cuda()
batch_ptr = torch.tensor([0, 5, 10]).long().cuda()

neighbors = torch.tensor([20, 21] + [3]).long().cuda()
neighbors_ptr = torch.tensor([0, 2, 3]).long().cuda()

for i in torch.ops.gs_ops.BatchCOOSlicing(0, coo_row, coo_col, batch_ptr,
                                          neighbors, neighbors_ptr):
    print(i)