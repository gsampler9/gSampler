#ifndef GS_HETERO_GRAP_H_
#define GS_HETERO_GRAP_H_

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <torch/custom_class.h>
#include <torch/script.h>
#include <string>
#include "./graph.h"

namespace gs {

struct HeteroGraphInternalCache {
  thrust::device_vector<int64_t*> all_indices, all_indptr;
};

struct NodeInfo {
  int64_t node_type;
  int64_t n_nodes;
  std::map<std::string, torch::Tensor> features;
};

struct EdgeRelation {
  int64_t src_type, dst_type;
  int64_t edge_type;
  c10::intrusive_ptr<Graph> homo_graph;
  std::map<std::string, torch::Tensor> features;
};

// A HeteroGraph wraps a graph with multiple nodes and edge types.
// For each node/edge type, we use the node_type_mapping_/edge_type_mapping_ to
// map it to an integer. Each mapped node/edge integer has a corresponding
// NodeInfo/EdgeRelation A NodeInfo describes the information of a node type,
// including its number and features. An EdgeRelation describes the node types
// of its destination and its homogeneous graph.
class HeteroGraph : public torch::CustomClassHolder {
 public:
  HeteroGraph() {}

  /**
   * @brief Build a HeteroGraph from a set of homogeneous graphs.
   *
   * @param node_types a set of node type and its size
   * @param edge_types a set of edge types with (src_type, dst_type, edge_type)
   * @param edge_ralations a set of homogeneous graphs describing the edges of
   * each edge type
   *
   */
  void LoadFromHomo(
      const std::vector<std::string>& node_types,
      const std::vector<std::tuple<std::string, std::string, std::string>>&
          edge_types,
      const std::vector<c10::intrusive_ptr<Graph>>& edge_relations);

  c10::intrusive_ptr<Graph> GetHomoGraph(const std::string& edge_type) const;

  torch::Tensor MetapathRandomWalkFused(
      torch::Tensor seeds, const std::vector<std::string>& metapath);

 private:
  int64_t n_node_types_, n_edge_types_;
  std::map<std::string, int64_t> node_type_mapping_, edge_type_mapping_;
  std::map<int64_t, EdgeRelation> hetero_edges_;
  std::map<int64_t, NodeInfo> hetero_nodes_;
  HeteroGraphInternalCache hg_cache_;
};

}  // namespace gs

#endif