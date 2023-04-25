
#include "./hetero_graph.h"
#include <thrust/device_vector.h>
#include "cuda/fusion/metapath_random_walk.h"
#include "cuda/graph_ops.h"

namespace gs {

/**
 * @brief Build a HeteroGraph from a set of homogeneous graphs.
 *
 * @param node_types a set of node type and its size
 * @param edge_types a set of edge types with (src_type, edge_type, dest_type)
 * @param edge_ralations a set of homogeneous graphs describing the edges of
 * each edge type
 *
 */

// TODO: initialize n_nodes in NodeInfo
void HeteroGraph::LoadFromHomo(
    const std::vector<std::string>& node_types,
    const std::vector<std::tuple<std::string, std::string, std::string>>&
        edge_types,
    const std::vector<c10::intrusive_ptr<Graph>>& edge_relations) {
  this->n_node_types_ = node_types.size();
  this->n_edge_types_ = edge_types.size();
  // Note: currently nodeInfo set n_nodes to zero
  for (int64_t i = 0; i < this->n_node_types_; i++) {
    this->node_type_mapping_.insert(std::make_pair(node_types.at(i), i));
    NodeInfo nodeInfo = {i, 0, {}};
    this->hetero_nodes_.insert(std::make_pair(i, nodeInfo));
  }
  thrust::host_vector<int64_t*> all_indices_host(this->n_edge_types_);
  thrust::host_vector<int64_t*> all_indptr_host(this->n_edge_types_);
  for (int64_t i = 0; i < this->n_edge_types_; i++) {
    edge_type_mapping_.insert(std::make_pair(std::get<1>(edge_types.at(i)), i));
    auto graph = edge_relations.at(i).get();
    EdgeRelation edge_relation = {
        this->node_type_mapping_[std::get<0>(edge_types.at(i))],
        this->node_type_mapping_[std::get<2>(edge_types.at(i))],
        this->edge_type_mapping_[std::get<1>(edge_types.at(i))],
        edge_relations.at(i),
        {}};
    hetero_edges_.insert(std::make_pair(i, edge_relation));

    all_indices_host[i] = graph->GetCSC()->indices.data_ptr<int64_t>();
    all_indptr_host[i] = graph->GetCSC()->indptr.data_ptr<int64_t>();
  }
  this->hg_cache_.all_indices = all_indices_host;
  this->hg_cache_.all_indptr = all_indptr_host;
}

c10::intrusive_ptr<Graph> HeteroGraph::GetHomoGraph(
    const std::string& edge_type) const {
  int64_t edge_id = edge_type_mapping_.at(edge_type);
  return hetero_edges_.at(edge_id).homo_graph;
}

torch::Tensor HeteroGraph::MetapathRandomWalkFused(
    torch::Tensor seeds, const std::vector<std::string>& metapath) {
  int64_t path_length = metapath.size();
  std::vector<int64_t> metapath_mapped(path_length);
  for (int64_t i = 0; i < path_length; i++) {
    metapath_mapped[i] = edge_type_mapping_[metapath[i]];
  }
  auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  torch::Tensor metapath_tensor =
      torch::from_blob(metapath_mapped.data(), path_length, opts)
          .to(torch::kCUDA);
  torch::Tensor paths = impl::fusion::FusedMetapathRandomWalkCUDA(
      seeds, metapath_tensor,
      thrust::raw_pointer_cast(this->hg_cache_.all_indices.data()),
      thrust::raw_pointer_cast(this->hg_cache_.all_indptr.data()));
  return paths;
}
}  // namespace gs