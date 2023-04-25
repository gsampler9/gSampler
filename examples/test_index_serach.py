import gs
import torch

origin_data = torch.randint(0, 100000, (500, )).unique().long().cuda()
index = torch.randint(0, 499, (100, )).long().cuda()
keys = origin_data[index]
search = torch.ops.gs_ops.index_search(origin_data, keys)
print(search.equal(index))