from gs import Graph, Matrix
import gs
import torch

A = Graph(False)
indptr = torch.LongTensor([0, 1, 1, 3, 4]).to('cuda:0')
indices = torch.LongTensor([4, 0, 1, 2]).to('cuda:0')

print('Test Case 1: +++++++++++++++++++')
column_ids = torch.LongTensor([2, 3]).to('cuda:0')
A._CAPI_load_csc(indptr, indices)
m = Matrix(A)
subA = m.columnwise_slicing(column_ids)
print(subA._graph._CAPI_metadata())
subA = subA.columnwise_sampling(2, True)
print(subA._graph._CAPI_metadata())
subA = m.fused_columnwise_slicing_sampling(column_ids, 2, True)
print(subA._graph._CAPI_metadata())

print('Test Case 2: +++++++++++++++++++')
column_ids = torch.LongTensor([1, 3]).to('cuda:0')
A._CAPI_load_csc(indptr, indices)
m = Matrix(A)
subA = m.columnwise_slicing(column_ids)
print(subA._graph._CAPI_metadata())
subA = subA.columnwise_sampling(2, True)
print(subA._graph._CAPI_metadata())
subA = m.fused_columnwise_slicing_sampling(column_ids, 2, True)
print(subA._graph._CAPI_metadata())


# def sampling(A: Matrix):
#     column_ids = torch.LongTensor([2, 3]).to('cuda:0')
#     return A[:, column_ids]


# m = Matrix(A)
# compiled_func = gs.jit.compile(func=sampling, args=(m, ))
# print("\nOrigin:")
# sampling(m)._graph.print()

# print()

# print("\nCompiled:")
# print(compiled_func(m, ))
# compiled_func(m, )._graph.print()
