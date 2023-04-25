#include "graph_ops.h"

#include "cuda_common.h"
#include "utils.h"

namespace gs {
namespace impl {

/*!
 * @brief Repeat elements.
 *
 * @param pos: The position of the output buffer to write the value.
 * @param out: Output buffer.
 * @param n_col: Length of positions
 * @param length: Number of values
 *
 * For example:
 * pos = [0, 1, 3, 4]
 * (implicit) val = [0, 1, 2]
 * then,
 * out = [0, 1, 1, 2]
 */
template <typename IdType, bool UseNMap>
__global__ void _RepeatKernel(const IdType* pos, const IdType* NIDMap,
                              IdType* out, int64_t n_col, int64_t length) {
  IdType tx = static_cast<IdType>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    IdType i = cub::UpperBound(pos, n_col, tx) - 1;
    out[tx] = UseNMap ? NIDMap[i] : i;
    tx += stride_x;
  }
}

std::pair<torch::Tensor, torch::Tensor> CSC2COOCUDA(torch::Tensor indptr,
                                                    torch::Tensor indices) {
  auto coo_size = indices.numel();
  auto col = torch::zeros(coo_size, indptr.options());

  dim3 block(128);
  dim3 grid((coo_size + block.x - 1) / block.x);
  _RepeatKernel<int64_t, false>
      <<<grid, block>>>(indptr.data_ptr<int64_t>(), nullptr,
                        col.data_ptr<int64_t>(), indptr.numel(), coo_size);
  return {indices, col};
}

std::pair<torch::Tensor, torch::Tensor> DCSC2COOCUDA(torch::Tensor indptr,
                                                     torch::Tensor indices,
                                                     torch::Tensor ids) {
  auto coo_size = indices.numel();
  auto col = torch::zeros(coo_size, indptr.options());

  dim3 block(128);
  dim3 grid((coo_size + block.x - 1) / block.x);
  _RepeatKernel<int64_t, true>
      <<<grid, block>>>(indptr.data_ptr<int64_t>(), ids.data_ptr<int64_t>(),
                        col.data_ptr<int64_t>(), indptr.numel(), coo_size);
  return {indices, col};
}

template <typename T>
int _NumberOfBits(const T& range) {
  if (range <= 1) {
    // ranges of 0 or 1 require no bits to store
    return 0;
  }

  int bits = 1;
  while (bits < static_cast<int>(sizeof(T) * 8) && (1 << bits) < range) {
    ++bits;
  }

  CHECK_EQ((range - 1) >> bits, 0);
  CHECK_NE((range - 1) >> (bits - 1), 0);

  return bits;
}

template <typename IdType>
inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> COOSort(
    torch::Tensor coo_key, torch::Tensor coo_value, int64_t num_keys) {
  auto num_items = coo_key.numel();
  const int num_bits = _NumberOfBits(num_keys);

  torch::Tensor input_key = coo_key;
  torch::Tensor input_value;
  torch::Tensor output_key = torch::zeros_like(coo_key);
  torch::Tensor output_value;

  input_value = torch::arange(num_items,
                              torch::dtype(torch::kInt64).device(torch::kCUDA));
  output_value = torch::zeros_like(input_value);
  cub_sortPairs<IdType, int64_t>(
      input_key.data_ptr<IdType>(), output_key.data_ptr<IdType>(),
      input_value.data_ptr<int64_t>(), output_value.data_ptr<int64_t>(),
      num_items, num_bits);

  return {output_key, coo_value.index({output_value}), output_value};
}

/*!
 * \brief Search for the insertion positions for needle in the hay.
 *
 * The hay is a list of sorted elements and the result is the insertion position
 * of each needle so that the insertion still gives sorted order.
 *
 * It essentially perform binary search to find upper bound for each needle
 * elements.
 *
 * For example:
 * hay = [0, 0, 1, 2, 2]
 * (implicit) needle = [0, 1, 2, 3]
 * then,
 * out = [2, 3, 5, 5]
 *
 * hay = [0, 0, 1, 3, 3]
 * (implicit) needle = [0, 1, 2, 3]
 * then,
 * out = [2, 3, 3, 5]
 */
template <typename IdType>
__global__ void _SortedSearchKernelUpperBound(const IdType* hay,
                                              int64_t hay_size,
                                              int64_t num_needles,
                                              IdType* pos) {
  IdType tx = static_cast<IdType>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < num_needles) {
    pos[tx] = cub::UpperBound(hay, hay_size, tx);
    tx += stride_x;
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
COO2CSCCUDA(torch::Tensor row, torch::Tensor col, int64_t num_cols,
            bool col_sorted) {
  torch::Tensor sort_row, sort_col;
  torch::optional<torch::Tensor> sort_index;
  if (col_sorted) {
    sort_col = col, sort_row = row;
    sort_index = torch::nullopt;
  } else {
    std::tie(sort_col, sort_row, sort_index) =
        COOSort<int64_t>(col, row, num_cols);
  }

  auto indptr = torch::zeros(num_cols + 1, sort_col.options());

  dim3 block(128);
  dim3 grid((num_cols + block.x - 1) / block.x);
  _SortedSearchKernelUpperBound<int64_t>
      <<<grid, block>>>(sort_col.data_ptr<int64_t>(), sort_col.numel(),
                        num_cols, indptr.data_ptr<int64_t>() + 1);
  return std::make_tuple(indptr, sort_row, sort_index);
}

template <typename IdType>
__global__ void _SortedSearchKernelUpperBoundWithMapping(const IdType* hay,
                                                         int64_t hay_size,
                                                         int64_t num_needles,
                                                         IdType* pos,
                                                         IdType* map) {
  IdType tx = static_cast<IdType>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < num_needles) {
    pos[tx] = cub::UpperBound(hay, hay_size, map[tx]);
    tx += stride_x;
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>,
           torch::Tensor>
COO2DCSCCUDA(torch::Tensor row, torch::Tensor col, int64_t max_num_cols,
             bool col_sorted) {
  torch::Tensor sort_row, sort_col;
  torch::optional<torch::Tensor> sort_index;
  if (col_sorted) {
    sort_col = col, sort_row = row;
    sort_index = torch::nullopt;
  } else {
    std::tie(sort_col, sort_row, sort_index) =
        COOSort<int64_t>(col, row, max_num_cols);
  }

  auto d_unique_res = torch::empty_like(col);
  auto d_num_selected_out = torch::empty(1, col.options());
  cub_consecutiveUnique<int64_t>(
      sort_col.data_ptr<int64_t>(), d_unique_res.data_ptr<int64_t>(),
      d_num_selected_out.data_ptr<int64_t>(), col.numel());
  auto val_col_ids = d_unique_res.index({torch::indexing::Slice(
      torch::indexing::None, d_num_selected_out.item<int64_t>())});

  auto id_size = val_col_ids.numel();
  auto indptr = torch::zeros(id_size + 1, sort_row.options());

  dim3 block(128);
  dim3 grid((id_size + block.x - 1) / block.x);
  _SortedSearchKernelUpperBoundWithMapping<int64_t><<<grid, block>>>(
      sort_col.data_ptr<int64_t>(), sort_col.numel(), id_size,
      indptr.data_ptr<int64_t>() + 1, val_col_ids.data_ptr<int64_t>());
  return std::make_tuple(indptr, sort_row, sort_index, val_col_ids);
}

}  // namespace impl
}  // namespace gs
