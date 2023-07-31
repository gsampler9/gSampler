#include <cooperative_groups.h>
#include <curand_kernel.h>
#include "../cuda_common.h"
#include "../utils.h"
#include "node2vec.h"

#define MAX(x, y) ((x > y) ? x : y)
#define TID (threadIdx.x + blockIdx.x * blockDim.x)
#define LTID (threadIdx.x)
#define LID (threadIdx.x % 32)
#define WID (threadIdx.x / 32)
#define BLOCK_SIZE 256

using namespace cooperative_groups;

namespace gs {
namespace impl {
namespace fusion {

template <typename NType, typename EType>
__device__ __inline__ bool BinarySearch(NType* ptr, EType degree,
                                        NType target) {
  EType tmp_degree = degree;
  NType* tmp_ptr = ptr;

  int itr = 0;
  while (itr < 50) {
    if (tmp_ptr[tmp_degree / 2] == target) {
      return true;
    } else if (tmp_ptr[tmp_degree / 2] < target) {
      tmp_ptr += tmp_degree / 2;
      if (tmp_degree == 1) {
        return false;
      }
      tmp_degree = tmp_degree - tmp_degree / 2;
    } else {
      tmp_degree = tmp_degree / 2;
    }
    if (tmp_degree == 0) {
      return false;
    }
    itr++;
  }
  return false;
}

template <typename NType, typename EType>
__device__ __inline__ bool CheckConnect(NType* graph_indice,
                                        EType* graph_indptr, NType src,
                                        NType dst) {
  EType degree = graph_indptr[src + 1] - graph_indptr[src];
  /*
  IdType item = cub::UpperBound(graph_indice + graph_indptr[src], degree, dst);
  if (item == degree) {
    return false;
  } else {
    return true;
  }
  */

  if (BinarySearch(graph_indice + graph_indptr[src], degree, dst)) {
    return true;
  }
  return false;
}

template <typename EType, typename NType>
__global__ void _Node2VecKernel(const NType* seed_data, const int64_t num_seeds,
                                const uint64_t max_num_steps,
                                NType* graph_indice, EType* graph_indptr,
                                NType* out_traces_data, double p, double q) {
  // init
  curandState rng;
  uint64_t rand_seed = 7777777;
  curand_init(rand_seed + TID, 0, 0, &rng);
  double max_scale = MAX(p, q);

  NType total_num_threads = blockDim.x * gridDim.x;
  for (NType idx = TID; idx < num_seeds; idx += total_num_threads) {
    NType curr = seed_data[idx];
    out_traces_data[0 * num_seeds + idx] = curr;
  }

  // begin node2vec
  for (NType idx = TID; idx < num_seeds; idx += total_num_threads) {
    coalesced_group warp = coalesced_threads();

    NType curr = out_traces_data[0 * num_seeds + idx];
    NType lastV = TID;
    for (int step_idx = 0; step_idx < max_num_steps; step_idx++) {
      coalesced_group active = coalesced_threads();

      NType pick = -1;
      if (curr != -1) {
        const EType in_row_start = graph_indptr[curr];
        const EType deg = graph_indptr[curr + 1] - graph_indptr[curr];

        if (deg == 0) {
          pick = -1;
        } else if (deg > 1) {
          NType outV;
          do {
            int y = (int)floor(curand_uniform(&rng) * max_scale);

            bool early_reject = (y >= MAX(max_scale, 1.0));
            if (early_reject) continue;

            int x = (int)floor(curand_uniform(&rng) * deg);
            bool early_accept = (y < MIN(MIN(p, q), 1.0));
            if (early_accept) {
              outV = graph_indice[in_row_start + x];
              break;
            }

            double h;
            outV = graph_indice[in_row_start + x];
            if (lastV == outV) {
              h = p;
            } else if (CheckConnect(graph_indice, graph_indptr, lastV, outV)) {
              h = q;
            } else {
              h = 1.0;
            }
            if (y < h) break;
          } while (true);
          pick = outV;
        } else {
          // deg == 1
          pick = graph_indice[in_row_start];
        }
      }
      lastV = curr;
      curr = pick;
      out_traces_data[(step_idx + 1) * num_seeds + idx] = pick;
    }
  }
}

torch::Tensor FusedNode2VecCUDA(torch::Tensor seeds, int64_t walk_length,
                                torch::Tensor indices, torch::Tensor indptr,
                                double p, double q) {
  ID_TYPE_SWITCH(indptr.scalar_type(), EType, {
    ID_TYPE_SWITCH(indices.scalar_type(), NType, {
      const NType* seed_data = seeds.data_ptr<NType>();
      const int64_t num_seeds = seeds.numel();
      const uint64_t max_num_steps = (uint64_t)walk_length;
      int64_t outsize = num_seeds * (max_num_steps + 1);
      torch::Tensor out_traces_tensor =
          torch::full(outsize, -1, indices.options().device(torch::kCUDA));

      NType* out_traces_data = out_traces_tensor.data_ptr<NType>();
      dim3 block(BLOCK_SIZE);
      dim3 grid(num_seeds / BLOCK_SIZE + 1);
      _Node2VecKernel<<<grid, block>>>(seeds.data_ptr<NType>(), num_seeds,
                                       max_num_steps, indices.data_ptr<NType>(),
                                       indptr.data_ptr<EType>(),
                                       out_traces_data, p, q);
      return out_traces_tensor.reshape({seeds.numel(), -1});
    });
  });
  return torch::Tensor();
}

}  // namespace fusion
}  // namespace impl
}  // namespace gs