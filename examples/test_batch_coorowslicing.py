import gs
import torch
import time
import numpy as np

# coo_col = torch.tensor([0,0,1,1] + [2,3]).long().cuda()
# coo_row = torch.tensor([1,2,2,3] + [3,0]).long().cuda()
# indices_ptr = torch.tensor([0, 4, 6]).long().cuda()
# selected = torch.tensor([0, 1, 3]).long().cuda()
# selected_ptr = torch.tensor([0, 2, 3]).long().cuda()

# A = gs.Graph(False)


# A._CAPI_load_coo(coo_row, coo_col)
# subg, row, col, coo_ptr = A._CAPI_batch_slicing(selected, 1, gs._COO, gs._COO, indices_ptr, selected_ptr)
# print("row:",row)
# print("col:",col)
# print("coo_ptr:",coo_ptr)

coo_row = torch.arange(64).long().cuda()
coo_col = torch.arange(0,64,2).repeat_interleave(2).long().cuda()
indices_ptr = torch.arange(0,65,2).long().cuda()
selected = torch.arange(0,63,2).long().cuda()
# 1 
selected_ptr = torch.arange(0,33,1).long().cuda()

A = gs.Graph(False)


A._CAPI_load_coo(coo_row, coo_col)
subg, row, col, coo_ptr = A._CAPI_batch_slicing(selected, 1, gs._COO, gs._COO, indices_ptr, selected_ptr)
print("row:",row)
print("col:",col)
print("coo_ptr:",coo_ptr)
