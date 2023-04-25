#include <torch/custom_class.h>
#include <torch/script.h>

#include "./graph.h"
#include "./graph_ops.h"
#include "./hetero_graph.h"
#include "./tensor_ops.h"
#include "cuda/tensor_ops.h"
using namespace gs;

TORCH_LIBRARY(gs_classes, m) {
  m.class_<Graph>("Graph")
      .def(torch::init<bool>())
      .def("_CAPI_SetData", &Graph::SetData)
      .def("_CAPI_GetData", &Graph::GetData)
      .def("_CAPI_FusionSlicing", &Graph::FusedBidirSlicing)
      .def("_CAPI_GetNumRows", &Graph::GetNumRows)
      .def("_CAPI_GetNumCols", &Graph::GetNumCols)
      .def("_CAPI_GetNumEdges", &Graph::GetNumEdges)
      .def("_CAPI_Slicing", &Graph::Slicing)
      .def("_CAPI_Sampling", &Graph::Sampling)
      .def("_CAPI_SamplingWithProbs", &Graph::SamplingProbs)
      .def("_CAPI_FusedColumnwiseSlicingSampling",
           &Graph::ColumnwiseFusedSlicingAndSampling)
      .def("_CAPI_LoadCSC", &Graph::LoadCSC)
      .def("_CAPI_LoadCOO", &Graph::LoadCOO)
      .def("_CAPI_LoadCSCWithColIds", &Graph::LoadCSCWithColIds)
      .def("_CAPI_AllValidNode", &Graph::AllValidNode)
      .def("_CAPI_GetRows", &Graph::GetRows)
      .def("_CAPI_GetCols", &Graph::GetCols)
      .def("_CAPI_GetValidRows", &Graph::GetValidRows)
      .def("_CAPI_GetValidCols", &Graph::GetValidCols)
      .def("_CAPI_GetCOORows", &Graph::GetCOORows)
      .def("_CAPI_GetCooCols", &Graph::GetCOOCols)
      .def("_CAPI_Relabel", &Graph::Relabel)
      .def("_CAPI_Sum", &Graph::Sum)
      .def("_CAPI_Normalize", &Graph::Normalize)
      .def("_CAPI_Divide", &Graph::Divide)
      .def("_CAPI_Metadata", &Graph::MetaData)
      .def("_CAPI_COOMetadata", &Graph::COOMetaData)
      .def("_CAPI_CSCMetadata", &Graph::CSCMetaData)
      .def("_CAPI_RandomWalk", &Graph::RandomWalk)
      .def("_CAPI_SDDMM", &Graph::SDDMM)
      .def("_CAPI_Split", &Graph::Split)
      .def("_CAPI_GetCSC", &Graph::GetCSCTensor)
      .def("_CAPI_GetCOO", &Graph::GetCOOTensor)
      .def("_CAPI_SetCOO", &Graph::SetCOOByTensor)
      .def("_CAPI_BatchSlicing", &Graph::BatchSlicing)
      .def("_CAPI_Decode", &Graph::Decode)
      .def("_CAPI_SetMetadata", &Graph::SetMetaData)
      .def("_CAPI_BatchFusionSlicing", &Graph::BatchFusedBidirSlicing)
      .def("_CAPI_EDivUSum", &Graph::EDivUSum);

  m.class_<HeteroGraph>("HeteroGraph")
      .def(torch::init<>())
      .def("_CAPI_LoadFromHomo", &HeteroGraph::LoadFromHomo)
      .def("_CAPI_GetHomoGraph", &HeteroGraph::GetHomoGraph)
      .def("_CAPI_MetapathRandomWalkFused",
           &HeteroGraph::MetapathRandomWalkFused);
}

TORCH_LIBRARY(gs_ops, m) {
  m.def("_CAPI_ListSampling", &ListSampling);
  m.def("_CAPI_ListSamplingWithProbs", &ListSamplingProbs);
  m.def("_CAPI_BatchListSamplingWithProbs", &BatchListSamplingProbs);
  m.def("_CAPI_IndexSearch", &IndexSearch);
  m.def("_CAPI_SplitByOffset", &SplitByOffset);
  m.def("_CAPI_IndptrSplitBySize", &gs::impl::SplitIndptrBySizeCUDA);
  m.def("_CAPI_IndptrSplitByOffset", &gs::impl::SplitIndptrByOffsetCUDA);
  m.def("_CAPI_BatchConcat", &gs::impl::BatchConcatCUDA);
  m.def("_CAPI_BatchUnique", &gs::impl::BatchUniqueCUDA);
  m.def("_CAPI_BatchUniqueByKey", &gs::impl::BatchUniqueByKeyCUDA);
  m.def("_CAPI_BatchUnique2", &gs::impl::BatchUnique2CUDA);
  m.def("_CAPI_BatchUniqueByKey2", &gs::impl::BatchUniqueByKey2CUDA);
  m.def("_CAPI_BatchCSRRelabelByKey", &gs::impl::BatchCSRRelabelByKeyCUDA);
  m.def("_CAPI_BatchCSRRelabel", &gs::impl::BatchCSRRelabelCUDA);
  m.def("_CAPI_BatchCOORelabelByKey", &gs::impl::BatchCOORelabelByKeyCUDA);
  m.def("_CAPI_BatchCOORelabel", &gs::impl::BatchCOORelabelCUDA);
  m.def("_CAPI_BatchSplit", &gs::impl::BatchSplit2CUDA);
  m.def("_CAPI_BatchCOOSlicing", &gs::impl::BatchCOOSlicingCUDA);
  m.def("_CAPI_BatchEncode", &gs::impl::BatchEncodeCUDA);
  m.def("_CAPI_BatchDecode", &gs::impl::BatchDecodeCUDA);
  m.def("_CAPI_GetBatchOffsets", &gs::impl::GetBatchOffsets);
  m.def("_CAPI_COORowSlicingGlobalId", &gs::impl::COORowSlicingGlobalIdCUDA);
  m.def("_CAPI_Unique", &gs::impl::TensorUniqueCUDA);
}

namespace gs {}