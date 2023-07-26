import torch
from ..utils import create_block_from_coo
from ..format import _COO, _CSC, _CSR
from ..ops import gspmm, gsddmm
from .matrix import Matrix, assign_block


class BatchMatrix(Matrix):

    def __init__(self,
                 graph=None,
                 row_ndata=None,
                 col_ndata=None,
                 edata=None,
                 batch_size=None,
                 num_batch=None):
        super().__init__(graph, row_ndata, col_ndata, edata, False)
        self.batch_size = batch_size
        self.num_batch = num_batch
        self.encoding = False

    def load_from_matrix(self, matrix, batch_size, num_batch):
        self.__init__(matrix._graph, matrix.row_ndata, matrix.col_ndata,
                      matrix.edata)
        self.batch_size = batch_size
        self.num_batch = num_batch

    def __getitem__(self, data):
        assert len(data) == 2
        r = data[0]
        c = data[1]
        assert isinstance(r, slice)
        assert isinstance(c, slice)
        print(r)
        print(c)

        r_slice = r.start
        r_slice_ptr = r.step
        c_slice = c.start
        c_slice_ptr = c.step

        ret_matrix = BatchMatrix()

        has_r = False if r_slice is None else True
        has_c = False if c_slice is None else True

        if has_r and not has_c:
            raise NotImplementedError
        elif not has_r and has_c:
            # only column slicing
            if "_ID" not in self.col_ndata:
                ret_matrix.col_ndata["_ID"] = c_slice

            graph, edge_index = self._graph._CAPI_BatchColSlicing(
                c_slice, c_slice_ptr, self.encoding)

            ret_matrix._graph = graph
            ret_matrix.num_batch = self.num_batch
            for key, value in self.edata.items():
                ret_matrix.edata[key] = value[edge_index]

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
        raise NotImplementedError

    def to_dgl_block(self, prefetch_edata={}):
        col_seeds = self.col_ndata.get("_ID", self.null_tensor)

        (frontier_list, row_list, col_list,
         eid_list) = self._graph._CAPI_BatchGraphRelabel(
             col_seeds, self.row_ndata.get("_ID", self.null_tensor))

        col_counts = self._graph._CAPI_BatchGetColCounts()

        blocks = []
        assign_edata = {key: self.edata[key] for key in prefetch_edata}
        for frontier, row, col, eid, col_count in zip(frontier_list, row_list,
                                                      col_list, eid_list,
                                                      col_counts):
            block = create_block_from_coo(row, col, frontier.numel(),
                                          col_count)
            assign_block(block, eid, assign_edata, frontier)
            blocks.append(block)

        return blocks

    def all_nodes(self) -> torch.Tensor:
        return self._graph._CAPI_BatchGetValidNodes(
            self.col_ndata.get("_ID", self.null_tensor),
            self.row_ndata.get("_ID", self.null_tensor),
        )

    def all_rows(self):
        return self._graph._CAPI_BatchGetCOORows()

    def all_cols(self):
        return self._graph._CAPI_BatchGetCOOCols()

    def all_edges(self):
        return self._graph._CAPI_BatchGetCOOEids()

    def random_walk(self, seeds, seeds_ptr, walk_length) -> torch.Tensor:
        return torch.ops.gs_ops._CAPI_BatchSplitByOffset(
            self._graph._CAPI_RandomWalk(seeds, walk_length), seeds_ptr)

    def node2vec(self, seeds, seeds_ptr, walk_length, p, q) -> torch.Tensor:
        return torch.ops.gs_ops._CAPI_BatchSplitByOffset(
            self._graph._CAPI_Node2Vec(seeds, walk_length, p, q), seeds_ptr)
