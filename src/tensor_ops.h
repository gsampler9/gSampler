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
std::tuple<torch::Tensor, torch::Tensor> ListSampling(torch::Tensor data,
                                                      int64_t num_picks,
                                                      bool replace);

std::tuple<torch::Tensor, torch::Tensor> ListSamplingProbs(torch::Tensor data,
                                                           torch::Tensor probs,
                                                           int64_t num_picks,
                                                           bool replace);

std::tuple<torch::Tensor, torch::Tensor> BatchListSamplingProbs(
    torch::Tensor probs, int64_t num_picks, bool replace, torch::Tensor range);

torch::Tensor IndexSearch(torch::Tensor origin_data, torch::Tensor keys);

std::vector<torch::Tensor> SplitByOffset(torch::Tensor data,
                                         torch::Tensor offset);
}  // namespace gs
#endif