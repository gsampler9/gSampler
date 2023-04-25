#include <gtest/gtest.h>
#include <torch/torch.h>
#include "graph.h"

using namespace gs;

TEST(SDDMM, add_test1)
{
    Graph A(false);
    auto Ioptions = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto Doptions = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor coo_row = torch::arange(0, 100, Ioptions);
    torch::Tensor coo_col = torch::arange(0, 100, Ioptions);
    torch::Tensor lhs = torch::arange(0, 100, Doptions);
    torch::Tensor rhs = torch::arange(0, 100, Doptions);
    torch::Tensor out = torch::zeros(100, Doptions);
    torch::Tensor res = torch::arange(0, 100, Doptions) * 2;
    A.LoadCOO(coo_row, coo_col);

    A.SDDMM("add", lhs, rhs, out, 0, 2, _COO);
    EXPECT_TRUE(res.equal(out));
}

TEST(SDDMM, add_test2)
{
    Graph A(false);
    auto Ioptions = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto Doptions = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, Ioptions) * 5;
    torch::Tensor indices = torch::arange(0, 100, Ioptions);
    torch::Tensor lhs = torch::arange(0, 100, Doptions);
    torch::Tensor rhs = torch::arange(0, 20, Doptions);
    torch::Tensor out = torch::zeros(100, Doptions);
    torch::Tensor res = lhs + torch::repeat_interleave(rhs, 5);
    A.LoadCSC(indptr, indices);

    A.SDDMM("add", lhs, rhs, out, 0, 2, _COO);
    EXPECT_TRUE(res.equal(out));
}

TEST(SDDMM, add_test3)
{
    Graph A(false);
    auto Ioptions = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto Doptions = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor coo_row = torch::arange(0, 100, Ioptions);
    torch::Tensor coo_col = torch::arange(0, 100, Ioptions);
    torch::Tensor lhs = torch::arange(0, 100, Doptions);
    torch::Tensor rhs = torch::arange(0, 100, Doptions);
    torch::Tensor out = torch::zeros(100, Doptions);
    torch::Tensor res = torch::arange(0, 100, Doptions) * 2;
    A.LoadCOO(coo_row, coo_col);

    A.SDDMM("add", lhs, rhs, out, 0, 2, _CSR);
    EXPECT_TRUE(res.equal(out));
}

TEST(SDDMM, add_test4)
{
    Graph A(false);
    auto Ioptions = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto Doptions = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor col_ids = torch::arange(2, 22, Ioptions);
    torch::Tensor indptr = torch::arange(0, 21, Ioptions) * 5;
    torch::Tensor indices = torch::arange(0, 100, Ioptions);
    torch::Tensor lhs = torch::arange(0, 100, Doptions);
    torch::Tensor rhs = torch::arange(0, 20, Doptions);
    torch::Tensor out = torch::zeros(100, Doptions);
    torch::Tensor res = lhs + torch::repeat_interleave(rhs, 5);
    A.LoadCSCWithColIds(col_ids, indptr, indices);

    A.SDDMM("add", lhs, rhs, out, 0, 2, _CSR);
    EXPECT_TRUE(res.equal(out));
}

TEST(SDDMM, add_test5)
{
    Graph A(false);
    auto Ioptions = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto Doptions = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor coo_row = torch::arange(0, 100, Ioptions);
    torch::Tensor coo_col = torch::arange(0, 100, Ioptions);
    torch::Tensor lhs = torch::arange(0, 100, Doptions);
    torch::Tensor rhs = torch::arange(0, 100, Doptions);
    torch::Tensor out = torch::zeros(100, Doptions);
    torch::Tensor res = torch::arange(0, 100, Doptions) * 2;
    A.LoadCOO(coo_row, coo_col);

    A.SDDMM("add", lhs, rhs, out, 0, 2, _CSC);
    EXPECT_TRUE(res.equal(out));
}

TEST(SDDMM, add_test6)
{
    Graph A(false);
    auto Ioptions = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto Doptions = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, Ioptions) * 5;
    torch::Tensor indices = torch::arange(0, 100, Ioptions);
    torch::Tensor lhs = torch::arange(0, 100, Doptions);
    torch::Tensor rhs = torch::arange(0, 20, Doptions);
    torch::Tensor out = torch::zeros(100, Doptions);
    torch::Tensor res = lhs + torch::repeat_interleave(rhs, 5);
    A.LoadCSC(indptr, indices);

    A.SDDMM("add", lhs, rhs, out, 0, 2, _CSC);
    EXPECT_TRUE(res.equal(out));
}