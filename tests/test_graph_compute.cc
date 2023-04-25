#include <gtest/gtest.h>
#include "graph.h"
#include <torch/torch.h>
#include <chrono>

using namespace gs;

TEST(GraphSum, test1)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto data_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    torch::Tensor expected = torch::ones(20, data_options) * 5;
    A.LoadCSC(indptr, indices);

    auto result = A.Sum(0, 1, _CSC);

    EXPECT_EQ(result.numel(), expected.numel());
    EXPECT_TRUE(result.equal(expected));
}

TEST(GraphSum, test2)
{
    Graph A(false);
    auto options = torch::dtype(torch::kInt64).device(torch::kCUDA);
    auto data_options = torch::dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    torch::Tensor data = torch::arange(1, 6, data_options);
    torch::Tensor expected = (data.sum()).repeat({20});
    A.LoadCSC(indptr, indices);
    A.SetData(data.repeat({20}));

    auto result = A.Sum(0, 1, _CSC);

    EXPECT_EQ(result.numel(), expected.numel());
    EXPECT_TRUE(result.equal(expected));
}

TEST(GraphSum, test3)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto data_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    torch::Tensor expected = torch::ones(20, data_options) * 5;
    A.LoadCSC(indptr, indices);

    auto result = A.Sum(0, 2, _CSC);

    EXPECT_EQ(result.numel(), expected.numel());
    EXPECT_TRUE(result.equal(expected));
}

TEST(GraphSum, test4)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto data_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    torch::Tensor data = torch::arange(1, 6, data_options);
    torch::Tensor expected = (data.pow(2).sum()).repeat({20});
    A.LoadCSC(indptr, indices);
    A.SetData(data.repeat({20}));

    auto result = A.Sum(0, 2, _CSC);

    EXPECT_EQ(result.numel(), expected.numel());
    EXPECT_TRUE(result.equal(expected));
}

TEST(GraphSum, test5)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto data_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor indptr = torch::cat({torch::zeros(10, options), torch::arange(0, 3, options) * 2});
    torch::Tensor indices = torch::arange(1, 4, 2, options).repeat({2});
    torch::Tensor data = torch::arange(3, 7, data_options);
    torch::Tensor expected = torch::cat({torch::arange(0, 9, 8, options), torch::arange(0, 11, 10, options), torch::zeros(8, options)});
    A.LoadCSC(indptr, indices);
    A.SetData(data);

    A.CSC2CSR();
    auto result = A.Sum(1, 1, _CSR);

    EXPECT_TRUE(result.equal(expected));
}

TEST(GraphSum, test6)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto data_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor indptr = torch::cat({torch::zeros(10, options), torch::arange(0, 3, options) * 2});
    torch::Tensor indices = torch::arange(1, 4, 2, options).repeat({2});
    torch::Tensor data = torch::arange(3, 7, data_options);
    torch::Tensor expected = torch::cat({torch::arange(0, 9, 8, options), torch::arange(0, 11, 10, options), torch::zeros(8, options)});
    A.LoadCSC(indptr, indices);
    A.SetData(data);

    auto result = A.Sum(1, 1, _COO);

    EXPECT_TRUE(result.equal(expected));
}

TEST(GraphSum, test7)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto data_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor n_ids = torch::arange(10, 12, options);
    torch::Tensor indptr = torch::cat({torch::zeros(10, options), torch::arange(0, 3, options) * 2});
    torch::Tensor indices = torch::arange(1, 4, 2, options).repeat({2});
    torch::Tensor data = torch::arange(3, 7, data_options);
    torch::Tensor expected = torch::cat({torch::arange(0, 9, 8, data_options), torch::arange(0, 11, 10, data_options), torch::zeros(8, data_options)});
    A.LoadCSC(indptr, indices);
    A.SetData(data);
    A.Slicing(n_ids, 0, _CSC, _CSC);

    auto result = A.Sum(1, 1, _CSR);

    EXPECT_TRUE(result.equal(expected));
}

TEST(GraphDiv, test1)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto data_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    torch::Tensor divisor = torch::ones(20, data_options) * 10;
    torch::Tensor expected = torch::ones(100, data_options) / 10;
    A.LoadCSC(indptr, indices);

    auto graph_ptr = A.Divide(divisor, 0, _CSC);
    auto result = graph_ptr->GetData().value();

    EXPECT_EQ(result.numel(), expected.numel());
    EXPECT_TRUE(result.isclose(expected).all().item<bool>());
}

TEST(GraphDiv, test2)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto data_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    torch::Tensor divisor = torch::arange(1, 21, data_options);
    torch::Tensor expected = torch::repeat_interleave(torch::ones(20, data_options) / divisor, 5);
    A.LoadCSC(indptr, indices);

    auto graph_ptr = A.Divide(divisor, 0, _CSC);
    auto result = graph_ptr->GetData().value();

    EXPECT_EQ(result.numel(), expected.numel());
    EXPECT_TRUE(result.isclose(expected).all().item<bool>());
}

TEST(GraphDiv, test3)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto data_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    torch::Tensor divisor = torch::arange(1, 21, data_options);
    torch::Tensor expected = torch::repeat_interleave(torch::ones(20, data_options) / divisor, 5);
    A.LoadCSC(indptr, indices);

    auto graph_ptr = A.Divide(divisor, 0, _COO);
    auto result = graph_ptr->GetData().value();

    EXPECT_EQ(result.numel(), expected.numel());
    EXPECT_TRUE(result.isclose(expected).all().item<bool>());
}

TEST(GraphNormalize, test1)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto data_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    torch::Tensor expected = torch::ones(100, data_options) / 5;
    A.LoadCSC(indptr, indices);

    auto sum_result = A.Sum(0, 1, _CSC);
    auto graph_ptr = A.Divide(sum_result, 0, _CSC);
    auto result = graph_ptr->GetData().value();

    EXPECT_EQ(result.numel(), expected.numel());
    EXPECT_TRUE(result.isclose(expected).all().item<bool>());
}

TEST(GraphNormalize, test2)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto data_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    torch::Tensor data = torch::arange(1, 6, data_options);
    torch::Tensor expected = (data / data.sum()).repeat({20});
    A.LoadCSC(indptr, indices);
    A.SetData(data.repeat({20}));

    auto sum_result = A.Sum(0, 1, _CSC);
    auto graph_ptr = A.Divide(sum_result, 0, _CSC);
    auto result = graph_ptr->GetData().value();

    EXPECT_EQ(result.numel(), expected.numel());
    EXPECT_TRUE(result.isclose(expected).all().item<bool>());
}

TEST(GraphNormalize, test3)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto data_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    torch::Tensor data = torch::arange(1, 6, data_options);
    torch::Tensor expected = (data / data.sum()).repeat({20});
    A.LoadCSC(indptr, indices);
    A.SetData(data.repeat({20}));

    auto sum_result = A.Sum(0, 1, _COO);
    auto graph_ptr = A.Divide(sum_result, 0, _COO);
    auto result = graph_ptr->GetData().value();

    EXPECT_EQ(result.numel(), expected.numel());
    EXPECT_TRUE(result.isclose(expected).all().item<bool>());
}

TEST(GraphNormalize, test4)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto data_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor n_ids = torch::arange(10, 12, options);
    torch::Tensor indptr = torch::cat({torch::zeros(10, options), torch::arange(0, 3, options) * 2});
    torch::Tensor indices = torch::arange(1, 4, 2, options).repeat({2});
    torch::Tensor data = torch::arange(3, 7, data_options);
    torch::Tensor expected = torch::flatten(data.reshape({2, 2}) / data.reshape({2, 2}).sum(0));
    A.LoadCSC(indptr, indices);
    A.SetData(data);

    auto graph_ptr = A.Slicing(n_ids, 0, _CSC, _CSC);
    auto sum_result = graph_ptr->Sum(1, 1, _CSR);
    graph_ptr = graph_ptr->Divide(sum_result, 1, _CSR);
    auto result = graph_ptr->GetData().value();

    EXPECT_TRUE(result.equal(expected));
}

TEST(GraphNormalize, test5)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto data_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor n_ids = torch::arange(10, 12, options);
    torch::Tensor indptr = torch::cat({torch::zeros(10, options), torch::arange(0, 3, options) * 2});
    torch::Tensor indices = torch::arange(1, 4, 2, options).repeat({2});
    torch::Tensor data = torch::arange(3, 7, data_options);
    torch::Tensor expected = torch::flatten(data.reshape({2, 2}) / data.reshape({2, 2}).sum(0));
    torch::Tensor divisor = torch::ones(2, data_options);
    A.LoadCSR(indptr, indices);
    A.SetData(data);
    auto graph_ptr = A.Slicing(n_ids, 1, _CSR, _CSR);

    graph_ptr = graph_ptr->Divide(divisor, 1, _CSR);
    auto sum_result = graph_ptr->Sum(0, 1, _CSC);
    graph_ptr = graph_ptr->Divide(sum_result, 0, _CSC);
    auto result = graph_ptr->GetData().value();

    EXPECT_TRUE(result.equal(expected));
}

TEST(GraphNormalize, test6)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto data_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor n_ids = torch::arange(10, 12, options);
    torch::Tensor indptr = torch::cat({torch::zeros(10, options), torch::arange(0, 3, options) * 2});
    torch::Tensor indices = torch::arange(1, 4, 2, options).repeat({2});
    torch::Tensor data = torch::arange(3, 7, data_options);
    torch::Tensor expected = torch::flatten(data.reshape({2, 2}) / data.reshape({2, 2}).sum(0));
    torch::Tensor divisor = torch::ones(2, data_options);
    A.LoadCSR(indptr, indices);
    A.SetData(data);
    auto graph_ptr = A.Slicing(n_ids, 1, _CSR, _CSR);

    graph_ptr = graph_ptr->Divide(divisor, 1, _CSR);
    auto sum_result = graph_ptr->Sum(0, 1, _COO);
    graph_ptr = graph_ptr->Divide(sum_result, 0, _COO);
    auto result = graph_ptr->GetData().value();

    EXPECT_TRUE(result.equal(expected));
}