#include <curand_kernel.h>
#include "../cuda_common.h"
#include "../utils.h"
#include "node2vec.h"

#define MAX(x, y) ((x > y) ? x : y)

namespace gs {
namespace impl {
namespace fusion {

template <typename IdType>
__device__ bool BinarySearch(IdType* ptr, IdType degree, IdType target) {
  IdType tmp_degree = degree;
  IdType* tmp_ptr = ptr;
  // printf("checking %d\t", target);
  IdType itr = 0;
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

template <typename IdType>
__inline__ __device__ bool CheckConnect(IdType* graph_indice,
                                        IdType* graph_indptr, IdType src,
                                        IdType dst) {
  IdType degree = graph_indptr[src + 1] - graph_indptr[src];
  IdType item = cub::UpperBound(graph_indice + graph_indptr[src], degree, dst);
  if (item == degree) {
    return false;
  } else {
    return true;
  }
  /*
  if (BinarySearch(graph_indice + graph_indptr[src], degree, dst)) {
    // paster()
    // printf("Connect %d %d \n", src, dst);
    return true;
  }
  // printf("not Connect %d %d \n", src, dst);
  return false;
  */
}

template <typename IdType>
__global__ void _Node2VecKernel(const IdType* seed_data,
                                const int64_t num_seeds,
                                const uint64_t max_num_steps,
                                IdType* graph_indice, IdType* graph_indptr,
                                IdType* out_traces_data, double p, double q) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t last_idx =
      min(static_cast<int64_t>(blockIdx.x + 1) * blockDim.x, num_seeds);
  double max_scale = MAX(p, q);
  for (int idx = tid; idx < last_idx; idx += blockDim.x) {
    IdType curr = seed_data[idx];
    out_traces_data[0 * num_seeds + idx] = curr;
  }
  curandState rng;
  uint64_t rand_seed = 7777777;
  curand_init(rand_seed + tid, 0, 0, &rng);
  int64_t grid_mul_block = gridDim.x * blockDim.x;
  // begin random walk
  IdType lastV = tid;
  for (int idx = tid; idx < num_seeds; idx += grid_mul_block) {
    for (int step_idx = 0; step_idx < max_num_steps; step_idx++) {
      IdType curr = out_traces_data[step_idx * num_seeds + idx];
      if (curr == -1) {
        out_traces_data[(step_idx + 1) * num_seeds + idx] = -1;
      } else {
        const IdType in_row_start = graph_indptr[curr];
        const IdType deg = graph_indptr[curr + 1] - graph_indptr[curr];
        if (deg > 1) {
          IdType outV;
          do {
            int64_t x = (int64_t)floor(curand_uniform(&rng) * deg);
            int64_t y = (int64_t)floor(curand_uniform(&rng) * max_scale);
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
          out_traces_data[(step_idx + 1) * num_seeds + idx] = outV;

        } else if (deg == 1) {
          out_traces_data[(step_idx + 1) * num_seeds + idx] =
              graph_indice[in_row_start];

        } else {
          out_traces_data[(step_idx + 1) * num_seeds + idx] = -1;
        }
      }
      lastV = curr;
    }
  }
}

torch::Tensor FusedNode2VecCUDA(torch::Tensor seeds, int64_t walk_length,
                                torch::Tensor indices, torch::Tensor indptr,
                                double p, double q) {
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
    _Node2VecKernel<<<nbx, ntx>>>(seeds.data_ptr<IdType>(), num_seeds,
                                  max_num_steps, indices.data_ptr<IdType>(),
                                  indptr.data_ptr<IdType>(), out_traces_data, p,
                                  q);
    return out_traces_tensor.reshape({seeds.numel(), -1});
  });

  return torch::Tensor();
}

}  // namespace fusion
}  // namespace impl
}  // namespace gs