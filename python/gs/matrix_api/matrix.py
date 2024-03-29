from __future__ import annotations
import torch
from torch.fx import Proxy
from dgl.heterograph import DGLBlock
from dgl.utils import gather_pinned_tensor_rows
from typing import Optional, List

from ..utils import create_block_from_coo, create_block_from_csc
from ..format import _COO, _CSC, _CSR
from ..ops import gspmm, gsddmm

torch.fx.wrap("create_block_from_coo")
torch.fx.wrap("create_block_from_csc")
torch.fx.wrap("assign_block")
torch.fx.wrap("gen_arange")
torch.fx.wrap("data_index")


def data_index(data, index):
    if data.is_pinned():
        return gather_pinned_tensor_rows(data, index)
    else:
        return data[index]


def gen_arange(size: int, dtype: torch.dtype,
               device: torch.device) -> torch.Tensor:
    return torch.arange(size, dtype=dtype, device=device)


def assign_block(block, e_ids, edata, unique_tensor):
    if e_ids is not None:
        block.edata["_ID"] = e_ids

    for key, value in edata.items():
        block.edata[key] = value

    block.srcdata["_ID"] = unique_tensor
    return block


class Matrix(object):

    def __init__(self,
                 graph=None,
                 row_ndata=None,
                 col_ndata=None,
                 edata=None,
                 compact=False):
        self._graph = graph
        self.null_tensor = torch.Tensor().cuda().long()
        self.row_ndata = {} if row_ndata is None else row_ndata
        self.col_ndata = {} if col_ndata is None else col_ndata
        self.edata = {} if edata is None else edata
        self._compact = compact

    def load_graph(self, format: str,
                   format_tensors: List[torch.Tensor]) -> Matrix:
        assert format in ["CSC", "COO", "CSR"]
        assert len(format_tensors) == 2

        if format == "COO":
            coo_row, coo_col = format_tensors
            num_rows = coo_row.max() + 1
            num_cols = coo_col.max() + 1
            self._graph = torch.classes.gs_classes.Graph(num_rows, num_cols)
            self._graph._CAPI_LoadCOO(coo_row, coo_col, False, False)

        elif format == "CSC":
            indptr, indices = format_tensors
            num_rows = indices.max() + 1
            num_cols = indptr.numel() - 1
            self._graph = torch.classes.gs_classes.Graph(num_rows, num_cols)
            self._graph._CAPI_LoadCSC(indptr, indices)

        elif format == "CSR":
            indptr, indices = format_tensors
            num_rows = indptr.numel() - 1
            num_cols = indices.max() + 1
            self._graph = torch.classes.gs_classes.Graph(num_rows, num_cols)
            self._graph._CAPI_LoadCSR(indptr, indices)

    # Extract-step operators
    def __getitem__(self, data) -> Matrix:
        assert len(data) == 2
        ret = self._graph
        r_slice = data[0]
        c_slice = data[1]

        ret_matrix = Matrix()

        edge_index = None
        row_index = None
        row_index_tag = False
        col_index = None
        col_index_tag = False
        graph = self._graph

        if isinstance(c_slice, Proxy) or isinstance(c_slice, torch.Tensor):
            if "_ID" not in self.col_ndata:
                ret_matrix.col_ndata["_ID"] = c_slice

            graph, _edge_index = graph._CAPI_Slicing(c_slice, 1, _CSC, _COO)
            edge_index = _edge_index

            col_index_tag = True
            col_index = c_slice

        if isinstance(r_slice, Proxy) or isinstance(r_slice, torch.Tensor):
            if "_ID" not in self.row_ndata:
                ret_matrix.row_ndata["_ID"] = r_slice

            graph, _edge_index = graph._CAPI_Slicing(r_slice, 0, _CSR, _COO)
            if edge_index is not None:
                edge_index = edge_index[_edge_index]
            else:
                edge_index = _edge_index

            row_index_tag = True
            row_index = r_slice

        ret_matrix._graph = graph
        for key, value in self.edata.items():
            ret_matrix.edata[key] = data_index(value, edge_index)

        for key, value in self.col_ndata.items():
            if col_index_tag:
                ret_matrix.col_ndata[key] = value[col_index]
            else:
                ret_matrix.col_ndata[key] = value

        for key, value in self.row_ndata.items():
            if row_index_tag:
                ret_matrix.row_ndata[key] = value[row_index]
            else:
                ret_matrix.row_ndata[key] = value

        if self._compact:
            ret_matrix.compact(0)
            ret_matrix.compact(1)

        return ret_matrix

    def compact(self, axis):
        if axis == 0:
            if "_ID" not in self.row_ndata:
                new_graph, unique_tensor = self._graph._CAPI_Compact(axis)
                for key, value in self.row_ndata.items():
                    self.row_ndata[key] = value[unique_tensor]
                self.row_ndata["_ID"] = unique_tensor

                self._graph = new_graph

        elif axis == 1:
            if "_ID" not in self.col_ndata:
                new_graph, unique_tensor = self._graph._CAPI_Compact(axis)
                for key, value in self.col_ndata.items():
                    self.col_ndata[key] = value[unique_tensor]
                self.col_ndata["_ID"] = unique_tensor

                self._graph = new_graph

        return self

    # Select-step operators
    def individual_sampling(self, K: int, probs: torch.Tensor,
                            replace: bool) -> Matrix:
        ret_matrix = Matrix()

        if probs is None:
            subgraph, edge_index = self._graph._CAPI_Sampling(
                1, K, replace, _CSC, _COO)
        else:
            subgraph, edge_index = self._graph._CAPI_SamplingProbs(
                1, probs, K, replace, _CSC, _COO)

        ret_matrix._graph = subgraph
        ret_matrix.row_ndata = self.row_ndata
        ret_matrix.col_ndata = self.col_ndata
        for key, value in self.edata.items():
            ret_matrix.edata[key] = value[edge_index]

        return ret_matrix

    def collective_sampling(self, K: int, probs: torch.Tensor,
                            replace: bool) -> Matrix:
        if probs is None:
            selected_index = torch.ops.gs_ops._CAPI_ListSampling(
                probs.numel(), K, replace)
        else:
            selected_index = torch.ops.gs_ops._CAPI_ListSamplingWithProbs(
                probs, K, replace)

        return self[selected_index, :], selected_index

    # Compute-step operators
    def sum(self, key, axis, on_format=_CSC) -> torch.Tensor:
        rhs = self.edata[key]
        if axis == 0:
            return gspmm(self, "copy_rhs", "sum", None, rhs, 0, _CSC)
        elif axis == 1:
            return gspmm(self, "copy_rhs", "sum", None, rhs, 2, _CSR)
        else:
            raise "axis should be 0 or 1"

        # Compute-step operators

    def div(self, key, divisor, axis) -> Matrix:
        ret_m = Matrix(self._graph, self.row_ndata, self.col_ndata)
        lhs = self.edata[key]
        if axis == 0:
            ret_data = gsddmm(ret_m, "div", lhs, divisor, "e", "v", _COO)
        elif axis == 1:
            ret_data = gsddmm(ret_m, "div", lhs, divisor, "e", "u", _COO)
        else:
            raise "axis should be 0 or 1"
        ret_m.edata = self.edata.copy()
        ret_m.edata[key] = ret_data
        return ret_m

    def to_dgl_block(self, prefetch_edata={}) -> DGLBlock:
        col_seeds = self.col_ndata.get("_ID", self.null_tensor)

        (
            unique_tensor,
            format_tensor1,
            format_tensor2,
            e_ids,
        ) = self._graph._CAPI_GraphRelabel(
            col_seeds,
            self.row_ndata.get("_ID", self.null_tensor),
        )

        block = create_block_from_coo(
            format_tensor1,
            format_tensor2,
            num_src=unique_tensor.numel(),
            num_dst=col_seeds.numel(),
        )

        assign_edata = {key: self.edata[key] for key in prefetch_edata}
        return assign_block(block, e_ids, assign_edata, unique_tensor)

    def all_nodes(self) -> torch.Tensor:
        return self._graph._CAPI_GetValidNodes(
            self.col_ndata.get("_ID", self.null_tensor),
            self.row_ndata.get("_ID", self.null_tensor),
        )

    def num_rows(self) -> int:
        return self._graph._CAPI_GetNumRows()

    def num_cols(self) -> int:
        return self._graph._CAPI_GetNumCols()

    def num_edges(self) -> int:
        return self._graph._CAPI_GetNumEdges()

    @property
    def csc(self) -> List[torch.Tensor]:
        return [
            self._graph._CAPI_GetCSCIndptr(),
            self._graph._CAPI_GetCSCIndices(),
            self._graph._CAPI_GetCSCEids(),
        ]

    @property
    def csr(self) -> List[torch.Tensor]:
        return [
            self._graph._CAPI_GetCSRIndptr(),
            self._graph._CAPI_GetCSRIndices(),
            self._graph._CAPI_GetCSREids(),
        ]

    @property
    def coo(self) -> List[torch.Tensor]:
        return [
            self._graph._CAPI_GetCOORows(),
            self._graph._CAPI_GetCOOCols(),
            self._graph._CAPI_GetCOOEids(),
        ]

    def random_walk(self, seeds, walk_length) -> torch.Tensor:
        return self._graph._CAPI_RandomWalk(seeds, walk_length)

    def node2vec(self, seeds, walk_length, p, q) -> torch.Tensor:
        return self._graph._CAPI_Node2Vec(seeds, walk_length, p, q)

    def rows(self) -> torch.Tensor:
        if "_ID" in self.row_ndata:
            return self.row_ndata["_ID"]
        else:
            return gen_arange(self.num_rows(),
                              dtype=torch.int64,
                              device='cuda')

    def cols(self) -> torch.Tensor:
        if "_ID" in self.col_ndata:
            return self.col_ndata["_ID"]
        else:
            return gen_arange(self.num_cols(),
                              dtype=torch.int64,
                              device='cuda')
