#include <curand_kernel.h>
#include "../cuda_common.h"
#include "../utils.h"
#include "node2vec.h"

#define MAX(x, y) ((x > y) ? x : y)

namespace gs {
namespace impl {
namespace fusion {

__device__ bool BinarySearch(int64_t* ptr, int64_t degree, int64_t target) {
  int64_t tmp_degree = degree;
  int64_t* tmp_ptr = ptr;
  // printf("checking %d\t", target);
  int64_t itr = 0;
  while (itr < 50) {
    // printf("%u %u.\t",tmp_ptr[tmp_degree / 2],target );
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

__device__ bool CheckConnect(int64_t* graph_indice, int64_t* graph_indptr,
                             int64_t degree, int64_t src, int64_t dst) {
  if (BinarySearch(graph_indice + graph_indptr[src], degree, dst)) {
    // paster()
    // printf("Connect %d %d \n", src, dst);
    return true;
  }
  // printf("not Connect %d %d \n", src, dst);
  return false;
}

__global__ void _Node2VecKernel(const int64_t* seed_data,
                                const int64_t num_seeds,
                                const uint64_t max_num_steps,
                                int64_t* graph_indice, int64_t* graph_indptr,
                                int64_t* out_traces_data, double p, double q) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t last_idx =
      min(static_cast<int64_t>(blockIdx.x + 1) * blockDim.x, num_seeds);
  double max_scale = MAX(p, q);
  for (int idx = tid; idx < last_idx; idx += blockDim.x) {
    int64_t curr = seed_data[idx];
    out_traces_data[0 * num_seeds + idx] = curr;
  }
  curandState rng;
  uint64_t rand_seed = 7777777;
  curand_init(rand_seed + tid, 0, 0, &rng);
  int64_t grid_mul_block = gridDim.x * blockDim.x;
  // begin random walk
  int64_t lastV = tid;
  for (int idx = tid; idx < num_seeds; idx += grid_mul_block) {
    for (int step_idx = 0; step_idx < max_num_steps; step_idx++) {
      int64_t curr = out_traces_data[step_idx * num_seeds + idx];
      int64_t pick = -1;
      if (curr == -1) {
        out_traces_data[(step_idx + 1) * num_seeds + idx] = pick;
      } else {
        const int64_t in_row_start = graph_indptr[curr];
        const int64_t deg = graph_indptr[curr + 1] - graph_indptr[curr];
        if (deg > 1) {
          int64_t outV;
          do {
            int64_t x = (int64_t)floor(curand_uniform(&rng) * deg);
            int64_t y = (int64_t)floor(curand_uniform(&rng) * max_scale);
            double h;
            outV = graph_indice[in_row_start + x];
            if (CheckConnect(graph_indice, graph_indptr, deg, curr, outV)) {
              h = q;
            } else if (lastV == outV) {
              h = p;
            } else {
              h = 1.0;
            }
            if (y < h) break;
          } while (true);
          out_traces_data[(step_idx + 1) * num_seeds + idx] = outV;
        } else if (deg == 1) {
          out_traces_data[(step_idx + 1) * num_seeds + idx] =
              graph_indice[in_row_start];
        } else {
          out_traces_data[(step_idx + 1) * num_seeds + idx] = -1;
        }
      }
    }
  }
}

torch::Tensor FusedNode2VecCUDA(torch::Tensor seeds, int64_t walk_length,
                                int64_t* indices, int64_t* indptr, double p,
                                double q) {
  const int64_t* seed_data = seeds.data_ptr<int64_t>();
  const int64_t num_seeds = seeds.numel();
  const uint64_t max_num_steps = (uint64_t)walk_length;
  int64_t outsize = num_seeds * (max_num_steps + 1);
  torch::Tensor out_traces_tensor = torch::empty(outsize, seeds.options());

  int64_t* out_traces_data = out_traces_tensor.data_ptr<int64_t>();
  const int ntx = FindNumThreads(num_seeds);
  dim3 block(ntx);
  const int nbx = (num_seeds + ntx - 1) / ntx;
  dim3 grid(nbx);
  _Node2VecKernel<<<nbx, ntx>>>(seeds.data_ptr<int64_t>(), num_seeds,
                                max_num_steps, indices, indptr, out_traces_data,
                                p, q);
  return out_traces_tensor.reshape({-1, seeds.numel()});
}

}  // namespace fusion
}  // namespace impl
}  // namespace gs