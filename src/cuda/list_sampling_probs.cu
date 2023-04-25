#include <curand_kernel.h>
#include <tuple>
#include "atomic.h"
#include "cuda_common.h"
#include "tensor_ops.h"
#include "utils.h"

namespace gs {
namespace impl {

template <typename IdType, typename FloatType>
std::tuple<torch::Tensor, torch::Tensor> _ListSamplingProbs(torch::Tensor data,
                                                            torch::Tensor probs,
                                                            int64_t num_picks,
                                                            bool replace) {
  int num_items = data.numel();
  torch::TensorOptions index_options =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);

  if (num_items <= num_picks && !replace) {
    // todo (ping), do we need clone here?
    return std::make_tuple(data, torch::arange(num_items, index_options));
  }

  torch::Tensor select;
  torch::Tensor index;

  if (replace) {
    // using cdf sampling
    torch::Tensor prefix_probs = probs.clone();
    select = torch::empty(num_picks, data.options());
    index = torch::empty(num_picks, index_options);

    // prefix_sum
    cub_inclusiveSum<FloatType>(prefix_probs.data_ptr<FloatType>(), num_items);

    uint64_t random_seed = 7777;
    using it = thrust::counting_iterator<IdType>;
    thrust::for_each(
        it(0), it(num_picks),
        [_in = data.data_ptr<IdType>(), _index = index.data_ptr<int64_t>(),
         _prefix_probs = prefix_probs.data_ptr<FloatType>(),
         _out = select.data_ptr<IdType>(), num_items,
         random_seed] __device__(IdType i) mutable {
          curandState rng;
          curand_init(i * random_seed, 0, 0, &rng);
          FloatType sum = _prefix_probs[num_items - 1];
          FloatType rand = static_cast<FloatType>(curand_uniform(&rng) * sum);
          int64_t item = cub::UpperBound<FloatType*, int64_t, FloatType>(
              _prefix_probs, num_items, rand);
          item = MIN(item, num_items - 1);
          // output
          _out[i] = _in[item];
          _index[i] = item;
        });
  } else {
    // using A-Res sampling
    torch::Tensor ares_tensor = torch::empty_like(probs);
    torch::Tensor ares_index = torch::empty(num_items, index_options);
    uint64_t random_seed = 7777;
    using it = thrust::counting_iterator<IdType>;
    thrust::for_each(it(0), it(num_items),
                     [_probs = probs.data_ptr<FloatType>(),
                      _ares = ares_tensor.data_ptr<FloatType>(),
                      _ares_ids = ares_index.data_ptr<int64_t>(),
                      random_seed] __device__(IdType i) mutable {
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

    cub_sortPairsDescending<FloatType, int64_t>(
        ares_tensor.data_ptr<FloatType>(), sort_ares.data_ptr<FloatType>(),
        ares_index.data_ptr<int64_t>(), sort_index.data_ptr<int64_t>(),
        num_items, 0);

    index = sort_index.slice(0, 0, num_picks, 1);
    select = data.index({index});
  }

  return std::make_tuple(select, index);
}

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

std::tuple<torch::Tensor, torch::Tensor> ListSamplingProbsCUDA(
    torch::Tensor data, torch::Tensor probs, int64_t num_picks, bool replace) {
  CHECK(data.dtype() == torch::kInt64);
  CHECK(probs.dtype() == torch::kFloat);
  assert(data.numel() == probs.numel());
  return _ListSamplingProbs<int64_t, float>(data, probs, num_picks, replace);
}

std::tuple<torch::Tensor, torch::Tensor> BatchListSamplingProbsCUDA(
    torch::Tensor probs, int64_t num_picks, bool replace, torch::Tensor range) {
  CHECK(probs.dtype() == torch::kFloat);
  return _BatchListSamplingProbs<int64_t, float>(probs, num_picks, replace,
                                                 range);
}

}  // namespace impl
}  // namespace gs