#include <torch/custom_class.h>
#include <torch/script.h>

#include "graph.h"
#include "graph_ops.h"
#include "tensor_ops.h"
using namespace gs;

TORCH_LIBRARY(gs_classes, m) {
  m.class_<Graph>("Graph")
      .def(torch::init<int64_t, int64_t>())
      .def("_CAPI_LoadCSC", &Graph::LoadCSC)
      .def("_CAPI_LoadCOO", &Graph::LoadCOO)
      .def("_CAPI_LoadCSR", &Graph::LoadCSR)
      .def("_CAPI_GetNumRows", &Graph::GetNumRows)
      .def("_CAPI_GetNumCols", &Graph::GetNumCols)
      .def("_CAPI_GetNumEdges", &Graph::GetNumEdges)
      .def("_CAPI_GetCSCIndptr", &Graph::GetCSCIndptr)
      .def("_CAPI_GetCSCIndices", &Graph::GetCSCIndices)
      .def("_CAPI_GetCSCEids", &Graph::GetCSCEids)
      .def("_CAPI_GetCOORows", &Graph::GetCOORows)
      .def("_CAPI_GetCOOCols", &Graph::GetCOOCols)
      .def("_CAPI_GetCOOEids", &Graph::GetCOOEids)
      .def("_CAPI_GetCSRIndptr", &Graph::GetCSRIndptr)
      .def("_CAPI_GetCSRIndices", &Graph::GetCSRIndices)
      .def("_CAPI_GetCSREids", &Graph::GetCSREids)
      .def("_CAPI_Slicing", &Graph::Slicing)
      .def("_CAPI_Sampling", &Graph::Sampling)
      .def("_CAPI_SamplingProbs", &Graph::SamplingProbs)
      .def("_CAPI_RandomWalk", &Graph::RandomWalk)
      .def("_CAPI_Node2Vec", &Graph::Node2Vec)
      .def("_CAPI_SDDMM", &Graph::SDDMM)
      .def("_CAPI_SpMM", &Graph::SpMM)
      .def("_CAPI_Compact", &Graph::Compact)
      .def("_CAPI_GraphRelabel", &Graph::GraphRelabel)
      .def("_CAPI_GetValidNodes", &Graph::GetValidNodes)
      .def("_CAPI_SlicingSampling", &Graph::FusedSlicingSampling)
      .def("_CAPI_FusedUOPV", &Graph::FusedUOPV)
      .def("_CAPI_FusedESquareSum", &Graph::FusedESquareSum)
      .def("_CAPI_FusedEDivUSum", &Graph::FusedEDivUSum)
      .def("_CAPI_BatchColSlicing", &Graph::BatchColSlicing)
      .def("_CAPI_BatchRowSampling", &Graph::BatchRowSampling)
      .def("_CAPI_BatchRowSamplingProbs", &Graph::BatchRowSamplingProbs)
      .def("_CAPI_BatchGraphRelabel", &Graph::BatchGraphRelabel)
      .def("_CAPI_GetEdgeBptr", &Graph::GetEdgeBptr)
      .def("_CAPI_GetColBptr", &Graph::GetColBptr)
      .def("_CAPI_BatchGetCSCIndptr", &Graph::BatchGetCSCIndptr)
      .def("_CAPI_BatchGetCSCIndices", &Graph::BatchGetCSCIndices)
      .def("_CAPI_BatchGetCSCEids", &Graph::BatchGetCSCEids)
      .def("_CAPI_BatchGetCOORows", &Graph::BatchGetCOORows)
      .def("_CAPI_BatchGetCOOCols", &Graph::BatchGetCOORows)
      .def("_CAPI_BatchGetCOOEids", &Graph::BatchGetCOOEids)
      .def("_CAPI_BatchGetColCounts", &Graph::BatchGetColCounts)
      .def("_CAPI_BatchGetValidNodes", &Graph::BatchGetValidNodes);
}

TORCH_LIBRARY(gs_ops, m) {
  m.def("_CAPI_ListSampling", &ListSampling);
  m.def("_CAPI_ListSamplingWithProbs", &ListSamplingProbs);
  m.def("_CAPI_BatchListSamplingWithProbs", &BatchListSamplingProbs);
  m.def("_CAPI_BatchListSampling", &BatchListSampling);
  m.def("_CAPI_BatchSplitByOffset", &gs::impl::batch::SplitByOffset);
}

namespace gs {}