#include <gtest/gtest.h>
#include "tensor_ops.h"
#include <torch/torch.h>

using namespace gs;

TEST(ListSampling, test1)
{
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    torch::Tensor data = torch::arange(1000, options).to(torch::kCUDA);
    torch::Tensor select;
    torch::Tensor index;

    std::tie(select, index) = ListSampling(data, 2000, false);
    EXPECT_TRUE(select.equal(data));
}

TEST(ListSampling, test2)
{
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    torch::Tensor data = torch::arange(1000, options).to(torch::kCUDA);
    torch::Tensor select;
    torch::Tensor index;

    std::tie(select, index) = ListSampling(data, 2000, true);
    ASSERT_TRUE(select.numel() == 2000);
    EXPECT_FALSE(select.equal(data));
    ASSERT_TRUE(std::get<0>(torch::_unique(select)).numel() <= 1000);
}

TEST(ListSampling, test3)
{
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    torch::Tensor data = torch::arange(1000, options).to(torch::kCUDA);
    torch::Tensor select;
    torch::Tensor index;

    std::tie(select, index) = ListSampling(data, 999, false);
    ASSERT_TRUE(select.numel() == 999);
    ASSERT_TRUE(std::get<0>(torch::_unique(select)).numel() == 999);
}