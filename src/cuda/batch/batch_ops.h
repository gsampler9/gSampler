#ifndef GS_CUDA_BATCH_OPS_H_
#define GS_CUDA_BATCH_OPS_H_

#include <torch/torch.h>

namespace gs {
namespace impl {
namespace batch {
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
BatchOnIndptrSlicingCUDA(torch::Tensor indptr, torch::Tensor indices,
                         torch::Tensor column_ids, torch::Tensor nid_ptr,
                         bool with_coo, bool encoding,
                         int64_t encoding_size = 0);

std::tuple<torch::Tensor, torch::Tensor> BatchListSamplingProbsCUDA(
    torch::Tensor probs, int64_t num_picks, bool replace, torch::Tensor range);

std::vector<torch::Tensor> SplitIndptrBySizeCUDA(torch::Tensor indptr,
                                                 int64_t size);

std::vector<torch::Tensor> SplitIndptrByOffsetCUDA(torch::Tensor indptr,
                                                   torch::Tensor offsets);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> BatchConcatCUDA(
    const std::vector<torch::Tensor> &data_tensors,
    const std::vector<torch::Tensor> &offset_tensors);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> BatchUniqueByKeyCUDA(
    torch::Tensor data_tensor, torch::Tensor data_ptr, torch::Tensor data_key);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> BatchUniqueCUDA(
    torch::Tensor data_tensor, torch::Tensor data_ptr);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> BatchUniqueByKey2CUDA(
    torch::Tensor data, torch::Tensor data_ptr, torch::Tensor data_key);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> BatchUnique2CUDA(
    torch::Tensor data, torch::Tensor data_ptr);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
BatchCSCRelabelByKeyCUDA(torch::Tensor seeds, torch::Tensor seeds_ptr,
                         torch::Tensor seeds_key, torch::Tensor indices,
                         torch::Tensor indices_ptr, torch::Tensor indices_key);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
BatchCSCRelabelCUDA(torch::Tensor seeds, torch::Tensor seeds_ptr,
                    torch::Tensor indices, torch::Tensor indices_ptr);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
BatchCOORelabelCUDA(torch::Tensor seeds, torch::Tensor seeds_ptr,
                    torch::Tensor coo_col, torch::Tensor coo_row,
                    torch::Tensor coo_ptr);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
BatchCOORelabelByKeyCUDA(torch::Tensor seeds, torch::Tensor seeds_ptr,
                         torch::Tensor seeds_key, torch::Tensor coo_col,
                         torch::Tensor coo_row, torch::Tensor coo_ptr,
                         torch::Tensor coo_key);

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>>
BatchSplit2CUDA(torch::Tensor data_tensor, torch::Tensor data_ptr_tensor,
                torch::Tensor data_key_tensor,
                const std::vector<torch::Tensor> &out_tensors,
                const std::vector<torch::Tensor> &out_ptr_tensors);

torch::Tensor BatchEncodeCUDA(torch::Tensor data_tensor, torch::Tensor data_ptr,
                              int64_t encoding_size);

torch::Tensor BatchDecodeCUDA(torch::Tensor in_data, int64_t encoding_size);

std::tuple<torch::Tensor, torch::Tensor> GetBatchOffsets(
    torch::Tensor data_tensor, int64_t num_batches, int64_t encoding_size);

std::vector<torch::Tensor> SplitByOffset(torch::Tensor data,
                                         torch::Tensor offset);
}  // namespace batch
}  // namespace impl
}  // namespace gs

#endif