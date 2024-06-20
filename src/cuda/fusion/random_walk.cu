#include <curand_kernel.h>
#include "../cuda_common.h"
#include "../utils.h"
#include "random_walk.h"

namespace gs {
namespace impl {
namespace fusion {

template <typename IdType>
__global__ void _RandomWalkKernel(const IdType* seed_data,
                                  const int64_t num_seeds,
                                  const uint64_t max_num_steps,
                                  IdType* graph_indice, IdType* graph_indptr,
                                  IdType* out_traces_data) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t last_idx =
      min(static_cast<int64_t>(blockIdx.x + 1) * blockDim.x, num_seeds);
  for (int idx = tid; idx < last_idx; idx += blockDim.x) {
    IdType curr = seed_data[idx];
    out_traces_data[0 * num_seeds + idx] = curr;
  }
  curandState rng;
  uint64_t rand_seed = 7777777;
  curand_init(rand_seed + tid, 0, 0, &rng);
  int64_t grid_mul_block = gridDim.x * blockDim.x;
  // begin random walk
  for (int idx = tid; idx < num_seeds; idx += grid_mul_block) {
    for (int step_idx = 0; step_idx < max_num_steps; step_idx++) {
      IdType curr = out_traces_data[step_idx * num_seeds + idx];

      IdType pick = -1;
      if (curr == -1) {
        out_traces_data[(step_idx + 1) * num_seeds + idx] = pick;
      } else {
        const IdType in_row_start = graph_indptr[curr];
        const IdType deg = graph_indptr[curr + 1] - graph_indptr[curr];
        if (deg > 0) {
          pick = graph_indice[in_row_start + curand(&rng) % deg];
          out_traces_data[(step_idx + 1) * num_seeds + idx] = pick;
        } else {
          out_traces_data[(step_idx + 1) * num_seeds + idx] = -1;
        }
      }
    }
  }
}

torch::Tensor FusedRandomWalkCUDA(torch::Tensor seeds, int64_t walk_length,
                                  torch::Tensor indices, torch::Tensor indptr) {
  ID_TYPE_SWITCH(indices.scalar_type(), IdType, {
    const IdType* seed_data = seeds.data_ptr<IdType>();
    const int64_t num_seeds = seeds.numel();
    const uint64_t max_num_steps = (uint64_t)walk_length;
    int64_t outsize = num_seeds * (max_num_steps + 1);
    torch::Tensor out_traces_tensor =
        torch::empty(outsize, indices.options().device(torch::kCUDA));

    IdType* out_traces_data = out_traces_tensor.data_ptr<IdType>();
    const int ntx = FindNumThreads(num_seeds);
    dim3 block(ntx);
    const int nbx = (num_seeds + ntx - 1) / ntx;
    dim3 grid(nbx);
    _RandomWalkKernel<<<nbx, ntx>>>(seeds.data_ptr<IdType>(), num_seeds,
                                    max_num_steps, indices.data_ptr<IdType>(),
                                    indptr.data_ptr<IdType>(), out_traces_data);
    return out_traces_tensor.reshape({-1, seeds.numel()});
  });
  return torch::Tensor();
}

}  // namespace fusion
}  // namespace impl
}  // namespace gs