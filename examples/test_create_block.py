import torch
from dgl import create_block
from gs.utils import create_block_from_coo, create_block_from_csc


def check_block(block1, block2):
    assert (block1.adj_sparse('coo')[0].equal(block2.adj_sparse('coo')[0]))
    assert (block1.adj_sparse('coo')[1].equal(block2.adj_sparse('coo')[1]))

    assert (block1.adj_sparse('csr')[0].equal(block2.adj_sparse('csr')[0]))
    assert (block1.adj_sparse('csr')[1].equal(block2.adj_sparse('csr')[1]))
    assert (block1.adj_sparse('csr')[2].equal(block2.adj_sparse('csr')[2]))

    assert (block1.adj_sparse('csc')[0].equal(block2.adj_sparse('csc')[0]))
    assert (block1.adj_sparse('csc')[1].equal(block2.adj_sparse('csc')[1]))
    assert (block1.adj_sparse('csc')[2].equal(block2.adj_sparse('csc')[2]))


print("creat_block from csc")
num_row = 4
num_col = 2
indptr = torch.tensor([0, 2, 3], device='cuda:0')
indices = torch.tensor([2, 3, 0], device='cuda:0')

print("dgl create_block")
csc_block1 = create_block(('csc', (indptr, indices, [])),
                          num_src_nodes=num_row,
                          num_dst_nodes=num_col)

print("our create_block")
csc_block2 = create_block_from_csc(indptr,
                                   indices,
                                   torch.tensor([]),
                                   num_src=num_row,
                                   num_dst=num_col)

print("check")
check_block(csc_block1, csc_block2)

print("creat_block from coo")
num_row = 4
num_col = 2
row = torch.tensor([2, 3, 0], device='cuda:0')
col = torch.tensor([0, 0, 1], device='cuda:0')

print("dgl create_block")
coo_block1 = create_block(('coo', (row, col)),
                          num_src_nodes=num_row,
                          num_dst_nodes=num_col)

print("our create_block")
coo_block2 = create_block_from_coo(row, col, num_src=num_row, num_dst=num_col)

print("check")
check_block(coo_block1, coo_block2)