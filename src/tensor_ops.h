#ifndef GS_TENSOR_OPS_H_
#define GS_TENSOR_OPS_H_

#include <torch/script.h>

namespace gs {

/**
 * @brief ListSampling, using A-Res sampling for replace = False and uniform
 * sampling for replace = True. Tt will return (selected_data, selected_index)
 *
 * @param data
 * @param num_picks
 * @param replace
 * @return std::tuple<torch::Tensor, torch::Tensor>
 */
torch::Tensor ListSampling(int64_t num_items, int64_t num_picks, bool replace);

torch::Tensor ListSamplingProbs(torch::Tensor probs, int64_t num_picks,
                                bool replace);
}  // namespace gs
#endif