#ifndef GS_TENSOR_OPS_H_
#define GS_TENSOR_OPS_H_

#include <torch/script.h>
#include "cuda/batch/batch_ops.h"

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

std::tuple<torch::Tensor, torch::Tensor> BatchListSamplingProbs(
    torch::Tensor probs, int64_t num_picks, bool replace, torch::Tensor range);

std::tuple<torch::Tensor, torch::Tensor> BatchListSampling(int64_t num_picks,
                                                           bool replace,
                                                           torch::Tensor range);
}  // namespace gs
#endif