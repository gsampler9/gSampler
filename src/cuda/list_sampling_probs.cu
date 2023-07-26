#include <curand_kernel.h>
#include <tuple>
#include "atomic.h"
#include "cuda_common.h"
#include "tensor_ops.h"
#include "utils.h"

namespace gs {
namespace impl {

template <typename IdType, typename FloatType>
torch::Tensor _ListSamplingProbs(torch::Tensor probs, int64_t num_picks,
                                 bool replace) {
  int num_items = probs.numel();
  torch::TensorOptions index_options =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);

  if (num_items <= num_picks && !replace) {
    // todo (ping), do we need clone here?
    return torch::arange(num_items, index_options);
  }

  torch::Tensor index;

  if (replace) {
    // using cdf sampling
    torch::Tensor prefix_probs = probs.clone();
    index = torch::empty(num_picks, index_options);

    // prefix_sum
    cub_inclusiveSum<FloatType>(prefix_probs.data_ptr<FloatType>(), num_items);

    uint64_t random_seed = 7777;
    using it = thrust::counting_iterator<IdType>;
    thrust::for_each(
        it(0), it(num_picks),
        [_index = index.data_ptr<int64_t>(),
         _prefix_probs = prefix_probs.data_ptr<FloatType>(), num_items,
         random_seed] __device__(IdType i) mutable {
          curandState rng;
          curand_init(i * random_seed, 0, 0, &rng);
          FloatType sum = _prefix_probs[num_items - 1];
          FloatType rand = static_cast<FloatType>(curand_uniform(&rng) * sum);
          int64_t item = cub::UpperBound<FloatType*, int64_t, FloatType>(
              _prefix_probs, num_items, rand);
          item = MIN(item, num_items - 1);
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
  }

  return index;
}

torch::Tensor ListSamplingProbsCUDA(torch::Tensor probs, int64_t num_picks,
                                    bool replace) {
  CHECK(probs.dtype() == torch::kFloat);
  return _ListSamplingProbs<int64_t, float>(probs, num_picks, replace);
}

}  // namespace impl
}  // namespace gs