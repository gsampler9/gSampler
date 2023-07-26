#include <curand_kernel.h>
#include <tuple>
#include "../atomic.h"
#include "../cuda_common.h"
#include "../utils.h"

namespace gs {
namespace impl {
namespace batch {
template <typename IdType>
__global__ void _BatchTensorSlicingKernel(int64_t* in_idx, int64_t* out_idx,
                                          int64_t* in_offsets,
                                          int64_t* out_offsets,
                                          int64_t num_batch) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
  while (row < num_batch) {
    int64_t in_startoff = in_offsets[row];
    int64_t in_endoff = in_offsets[row + 1];
    int64_t out_startoff = out_offsets[row];
    int64_t out_endoff = out_offsets[row + 1];
    int64_t size = out_endoff - out_startoff;
    for (int64_t idx = threadIdx.x; idx < size; idx += blockDim.x) {
      out_idx[out_startoff + idx] = in_idx[in_startoff + idx];
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType, typename FloatType>
std::tuple<torch::Tensor, torch::Tensor> _BatchListSamplingProbs(
    torch::Tensor probs, int64_t num_picks, bool replace, torch::Tensor range) {
  torch::Tensor out_idx, out_range = torch::empty_like(range);
  auto idx_options = torch::dtype(torch::kInt64).device(torch::kCUDA);
  int64_t num_split = range.numel() - 1;
  int64_t total_element = probs.numel();

  if (replace) {
    LOG(FATAL) << "Not implemented warning";
    // using it = thrust::counting_iterator<IdType>;
    // thrust::for_each(
    //     it(0), it(num_split + 1),
    //     [_out = out_range.data_ptr<int64_t>(),
    //      num_picks] __device__(IdType i) mutable { _out[i] = i * num_picks;
    //      });
    // out_data = torch::empty(num_picks * num_split, data.options());
    // out_idx = torch::empty(num_picks * num_split, idx_options);
  } else {
    using it = thrust::counting_iterator<IdType>;
    thrust::for_each(
        it(0), it(num_split),
        [_in = range.data_ptr<int64_t>(), _out = out_range.data_ptr<int64_t>(),
         num_picks] __device__(int64_t i) mutable {
          if (_in[i + 1] - _in[i] >= num_picks) {
            _out[i] = num_picks;
          } else {
            _out[i] = _in[i + 1] - _in[i];
          }
        });
    cub_exclusiveSum<IdType>(out_range.data_ptr<int64_t>(), num_split + 1);
    thrust::device_ptr<int64_t> item_prefix(out_range.data_ptr<int64_t>());
    int64_t out_length = item_prefix[num_split];  // cpu
    out_idx = torch::empty(out_length, idx_options);

    // using A-Res sampling
    torch::Tensor ares_tensor = torch::empty_like(probs);
    torch::Tensor ares_index = torch::empty(total_element, idx_options);
    uint64_t random_seed = 7777;
    using it = thrust::counting_iterator<IdType>;
    thrust::for_each(it(0), it(total_element),
                     [_probs = probs.data_ptr<FloatType>(),
                      _ares = ares_tensor.data_ptr<FloatType>(),
                      _ares_ids = ares_index.data_ptr<int64_t>(),
                      random_seed] __device__(int64_t i) mutable {
                       curandState rng;
                       curand_init(i * random_seed, 0, 0, &rng);
                       FloatType item_prob = _probs[i];
                       FloatType ares_prob =
                           item_prob <= 0
                               ? -1
                               : __powf(curand_uniform(&rng), 1.0f / item_prob);
                       _ares[i] = ares_prob;
                       _ares_ids[i] = i;
                     });
    torch::Tensor sort_ares = torch::empty_like(ares_tensor);
    torch::Tensor sort_index = torch::empty_like(ares_index);

    cub_segmentedSortPairsDescending<FloatType, int64_t>(
        ares_tensor.data_ptr<FloatType>(), sort_ares.data_ptr<FloatType>(),
        ares_index.data_ptr<int64_t>(), sort_index.data_ptr<int64_t>(),
        range.data_ptr<int64_t>(), total_element, num_split);

    dim3 block(16, 32);
    dim3 grid((num_split + block.y - 1) / block.y);
    CUDA_KERNEL_CALL((_BatchTensorSlicingKernel<IdType>), grid, block,
                     sort_index.data_ptr<int64_t>(),
                     out_idx.data_ptr<int64_t>(), range.data_ptr<int64_t>(),
                     out_range.data_ptr<int64_t>(), num_split);
  }
  return {out_idx, out_range};
}

std::tuple<torch::Tensor, torch::Tensor> BatchListSamplingProbsCUDA(
    torch::Tensor probs, int64_t num_picks, bool replace, torch::Tensor range) {
  CHECK(probs.dtype() == torch::kFloat);
  return _BatchListSamplingProbs<int64_t, float>(probs, num_picks, replace,
                                                 range);
}
}  // namespace batch
}  // namespace impl
}  // namespace gs