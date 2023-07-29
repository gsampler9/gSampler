import torch
from ..utils import create_block_from_coo
from ..format import _COO, _CSC, _CSR
from ..ops import gspmm, gsddmm
from .matrix import Matrix, data_index
from dgl.utils import gather_pinned_tensor_rows

torch.fx.wrap("batch_gen_block")
torch.fx.wrap("gather_pinned_tensor_rows")
torch.fx.wrap("data_index")


def batch_gen_block(frontier_list, coo_row_list, coo_col_list, coo_eids,
                    coo_ptr, edata, col_counts):
    edata_list = {}
    if coo_eids is not None:
        eids_list = torch.ops.gs_ops._CAPI_BatchSplitByOffset(
            coo_eids, coo_ptr)
        edata_list["_ID"] = eids_list

    for key, value in edata.items():
        edata_list[key] = torch.ops.gs_ops._CAPI_BatchSplitByOffset(
            value, coo_ptr)

    block_list = []
    for i in range(len(frontier_list)):
        block = create_block_from_coo(coo_row_list[i], coo_col_list[i],
                                      frontier_list[i].numel(), col_counts[i])

        block.srcdata["_ID"] = frontier_list[i]
        for key, value in edata_list.items():
            block.edata[key] = value[i]

        block_list.append(block)

    return block_list


class BatchMatrix(Matrix):

    def __init__(self,
                 graph=None,
                 row_ndata=None,
                 col_ndata=None,
                 edata=None,
                 encoding=True):
        super().__init__(graph, row_ndata, col_ndata, edata, False)
        self.encoding = encoding

    def load_from_matrix(self, matrix, encoding=True):
        self.__init__(matrix._graph, matrix.row_ndata, matrix.col_ndata,
                      matrix.edata)
        self.encoding = encoding

    def __getitem__(self, data):
        assert len(data) == 2
        r = data[0]
        c = data[1]
        assert isinstance(r, slice)
        assert isinstance(c, slice)

        r_slice = r.start
        r_slice_ptr = r.step
        c_slice = c.start
        c_slice_ptr = c.step

        ret_matrix = BatchMatrix()

        if (r_slice is not None
                and r_slice_ptr is not None) and (c_slice is not None
                                                  and c_slice_ptr is not None):
            graph, edge_index = self._graph._CAPI_BatchColRowSlcing(
                c_slice, c_slice_ptr, r_slice, r_slice_ptr)

            ret_matrix._graph = graph
            if "_ID" not in self.row_ndata:
                ret_matrix.row_ndata["_ID"] = r_slice
            else:
                raise NotImplementedError

            if "_ID" not in self.col_ndata:
                ret_matrix.col_ndata["_ID"] = c_slice
            else:
                raise NotImplementedError

            for key, value in self.edata.items():
                ret_matrix.edata[key] = data_index(value, edge_index)

            for key, value in self.row_ndata.items():
                ret_matrix.row_ndata[key] = value[r_slice]

            for key, value in self.col_ndata.items():
                ret_matrix.col_ndata[key] = value[c_slice]

            return ret_matrix

        elif (r_slice is not None and r_slice_ptr
              is not None) and (c_slice is None and c_slice_ptr is None):
            graph, edge_index = self._graph._CAPI_BatchRowSlicing(
                r_slice, r_slice_ptr)

            ret_matrix._graph = graph
            # only column slicing
            if "_ID" not in self.row_ndata:
                ret_matrix.row_ndata["_ID"] = r_slice

            for key, value in self.edata.items():
                ret_matrix.edata[key] = data_index(value, edge_index)

            for key, value in self.row_ndata.items():
                ret_matrix.row_ndata[key] = value[r_slice]

            for key, value in self.col_ndata.items():
                ret_matrix.col_ndata[key] = value

            return ret_matrix

        elif (c_slice is not None and c_slice_ptr
              is not None) and (r_slice is None and r_slice_ptr is None):

            if self.encoding:
                graph, edge_index = self._graph._CAPI_BatchColSlicing(
                    c_slice, c_slice_ptr, self.encoding)

                # only column slicing
                if "_ID" not in self.col_ndata:
                    ret_matrix.col_ndata["_ID"] = c_slice
                else:
                    raise NotImplementedError

                row_ids = None
                if "_ID" not in self.row_ndata:
                    row_ids, row_bptr = graph._CAPI_BatchGetRows()
                    ret_matrix.row_ndata["_ID"] = row_ids
                else:
                    raise NotImplementedError

                ret_matrix._graph = graph
                for key, value in self.edata.items():
                    ret_matrix.edata[key] = data_index(value, edge_index)

                for key, value in self.col_ndata.items():
                    ret_matrix.col_ndata[key] = value[c_slice]

                for key, value in self.row_ndata.items():
                    ret_matrix.row_ndata[key] = value

                return ret_matrix

            else:
                graph, edge_index = self._graph._CAPI_BatchColSlicing(
                    c_slice, c_slice_ptr, self.encoding)

                ret_matrix._graph = graph
                # only column slicing
                if "_ID" not in self.col_ndata:
                    ret_matrix.col_ndata["_ID"] = c_slice
                else:
                    raise NotImplementedError

                ret_matrix._graph = graph
                for key, value in self.edata.items():
                    ret_matrix.edata[key] = data_index(value, edge_index)

                for key, value in self.col_ndata.items():
                    ret_matrix.col_ndata[key] = value[c_slice]

                for key, value in self.row_ndata.items():
                    ret_matrix.row_ndata[key] = value

                return ret_matrix

        else:
            raise NotImplementedError

    def individual_sampling(self, K, probs, replace):
        ret_matrix = BatchMatrix()

        if probs is None:
            subgraph, edge_index = self._graph._CAPI_BatchRowSampling(
                K, replace)
        else:
            subgraph, edge_index = self._graph._CAPI_BatchRowSamplingWithProb(
                probs, K, replace)

        ret_matrix._graph = subgraph
        ret_matrix.row_ndata = self.row_ndata
        ret_matrix.col_ndata = self.col_ndata
        for key, value in self.edata.items():
            ret_matrix.edata[key] = value[edge_index]

        return ret_matrix

    def collective_sampling(self, K, probs, probs_ptr, replace):
        if probs is None:
            raise NotImplementedError
        else:
            selected_index, index_ptr = torch.ops.gs_ops._CAPI_BatchListSamplingWithProbs(
                probs, K, replace, probs_ptr)

            return self[selected_index::index_ptr, :], selected_index

    def to_dgl_block(self, prefetch_edata={}):
        (frontiers, frontiers_ptr, coo_row, coo_col, coo_eids,
         coo_bptr) = self._graph._CAPI_BatchGraphRelabel(
             self.col_ndata.get("_ID", self.null_tensor),
             self.row_ndata.get("_ID", self.null_tensor))

        frontier_list = torch.ops.gs_ops._CAPI_BatchSplitByOffset(
            frontiers, frontiers_ptr)

        coo_row_list = torch.ops.gs_ops._CAPI_BatchSplitByOffset(
            coo_row, coo_bptr)
        coo_col_list = torch.ops.gs_ops._CAPI_BatchSplitByOffset(
            coo_col, coo_bptr)
        col_counts = self._graph._CAPI_BatchGetColCounts()

        assign_edata = {key: self.edata[key] for key in prefetch_edata}

        return batch_gen_block(frontier_list, coo_row_list, coo_col_list,
                               coo_eids, coo_bptr, assign_edata, col_counts)

    def sum(self, key, axis) -> torch.Tensor:
        rhs = self.edata[key]
        if axis == 0:
            return gspmm(self, "copy_rhs", "sum", None, rhs, 0, _COO)
        elif axis == 1:
            return gspmm(self, "copy_rhs", "sum", None, rhs, 2, _COO)
        else:
            raise "axis should be 0 or 1"

    def div(self, key, divisor, axis) -> Matrix:
        ret_m = BatchMatrix(self._graph, self.row_ndata, self.col_ndata,
                            self.edata.copy(), self.encoding)
        lhs = self.edata[key]
        if axis == 0:
            ret_data = gsddmm(ret_m, "div", lhs, divisor, "e", "v", _COO)
        elif axis == 1:
            ret_data = gsddmm(ret_m, "div", lhs, divisor, "e", "u", _COO)
        else:
            raise "axis should be 0 or 1"
        ret_m.edata[key] = ret_data
        return ret_m

    def all_nodes(self) -> torch.Tensor:
        return self._graph._CAPI_BatchGetValidNodes(
            self.col_ndata.get("_ID", self.null_tensor),
            self.row_ndata.get("_ID", self.null_tensor),
        )

    def all_rows(self):
        return self._graph._CAPI_BatchGetRows()

    def all_cols(self):
        return self._graph._CAPI_BatchGetCols()

    def random_walk(self, seeds, seeds_ptr, walk_length) -> torch.Tensor:
        return torch.ops.gs_ops._CAPI_BatchSplitByOffset(
            self._graph._CAPI_RandomWalk(seeds, walk_length), seeds_ptr)

    def node2vec(self, seeds, seeds_ptr, walk_length, p, q) -> torch.Tensor:
        return torch.ops.gs_ops._CAPI_BatchSplitByOffset(
            self._graph._CAPI_Node2Vec(seeds, walk_length, p, q), seeds_ptr)
