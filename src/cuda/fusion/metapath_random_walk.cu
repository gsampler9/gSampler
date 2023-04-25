#include <curand_kernel.h>
#include "../cuda_common.h"
#include "../utils.h"
#include "metapath_random_walk.h"
namespace gs {
namespace impl {
namespace fusion {

template <int BLOCK_SIZE>
__global__ void _RandomWalkKernel(const int64_t* seed_data,
                                  const int64_t num_seeds,
                                  const int64_t* metapath_data,
                                  const uint64_t max_num_steps,
                                  int64_t** all_indices, int64_t** all_indptr,
                                  int64_t* out_traces_data) {
  int64_t tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int64_t last_idx =
      min(static_cast<int64_t>(blockIdx.x + 1) * BLOCK_SIZE, num_seeds);
  curandState rng;
  uint64_t rand_seed = 7777777;
  curand_init(rand_seed + tid, 0, 0, &rng);
  // fisrt step
  for (int idx = tid; idx < last_idx; idx += BLOCK_SIZE) {
    int64_t curr = seed_data[idx];
    out_traces_data[0 * num_seeds + idx] = curr;
  }
  // begin random walk
  for (int step_idx = 0; step_idx < max_num_steps; step_idx++) {
    for (int idx = tid; idx < last_idx; idx += BLOCK_SIZE) {
      int64_t curr = out_traces_data[step_idx * num_seeds + idx];
      int64_t pick = -1;
      if (curr < 0) {
        out_traces_data[(step_idx + 1) * num_seeds + idx] = pick;
      } else {
        int64_t metapath_id = metapath_data[step_idx];
        int64_t* graph_indice = all_indices[metapath_id];
        int64_t* graph_indptr = all_indptr[metapath_id];
        const int64_t in_row_start = graph_indptr[curr];
        const int64_t deg = graph_indptr[curr + 1] - graph_indptr[curr];
        if (deg > 0) {
          pick = graph_indice[in_row_start + curand(&rng) % deg];
        }
        out_traces_data[(step_idx + 1) * num_seeds + idx] = pick;
      }
    }
  }
}

torch::Tensor FusedMetapathRandomWalkCUDA(torch::Tensor seeds,
                                          torch::Tensor metapath,
                                          int64_t** all_indices,
                                          int64_t** all_indptr) {
  const int64_t* seed_data = seeds.data_ptr<int64_t>();
  const int64_t num_seeds = seeds.numel();
  const int64_t* metapath_data = metapath.data_ptr<int64_t>();
  const uint64_t max_num_steps = metapath.numel();
  int64_t outsize = num_seeds * (max_num_steps + 1);
  torch::Tensor out_traces_tensor = torch::empty(outsize, seeds.options());
  int64_t* out_traces_data = out_traces_tensor.data_ptr<int64_t>();
  constexpr int BLOCK_SIZE = 256;
  dim3 block(BLOCK_SIZE);
  dim3 grid((num_seeds + BLOCK_SIZE - 1) / BLOCK_SIZE);
  _RandomWalkKernel<BLOCK_SIZE><<<grid, block>>>(
      seeds.data_ptr<int64_t>(), num_seeds, metapath_data, max_num_steps,
      all_indices, all_indptr, out_traces_data);
  return out_traces_tensor.reshape({seeds.numel(), -1});
}
}  // namespace fusion
}  // namespace impl
}  // namespace gs