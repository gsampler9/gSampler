
#include <gtest/gtest.h>
#include "graph.h"
#include <torch/torch.h>

using namespace gs;

// function to check if there is an edge in homograph
static bool check_if_edge_exist(int64_t src, int64_t dst, int64_t *indptr_arr, int64_t *indices_arr)
{
    // if the dest is -1, directly return true by default
    if (dst == -1)
    {
        return true;
    }
    int64_t begin = indptr_arr[src];
    int64_t end = indptr_arr[src + 1];
    for (int64_t i = begin; i < end; i++)
    {
        if (indices_arr[i] == dst)
        {
            return true;
        }
    }
    return false;
}

TEST(RandomWalk, testGraphWithMultiPath)
{
    // homo graph1(user follow user): 0->1, 1->2, 2->3
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    int64_t indptr1[] = {0, 0, 1, 2, 3, 5};
    int64_t indices1[] = {0, 1, 1, 2, 3};
    auto indptr = torch::from_blob(indptr1, {5}, options).to(torch::kCUDA);
    auto indices = torch::from_blob(indices1, {5}, options).to(torch::kCUDA);
    int64_t walk_length = 3;
    Graph A(indptr.numel() - 1, indptr.numel() - 1);
    A.LoadCSC(indptr, indices);

    std::vector<int64_t> seeds_vector = {4, 2, 3, 1};
    torch::Tensor seeds = torch::from_blob(seeds_vector.data(), {4}, options).to(torch::kCUDA);
    torch::Tensor actual_path_tensor = A.RandomWalk(seeds, walk_length).reshape({-1, seeds.numel()}).to(torch::kCPU);
    /*
    if seed node is 4, there are two possible path: 4->2->1->0, 4->3->1->0,
    if seed node is 1 and 2 there are two possible path: 3->1->-0->-1, 2->1->0->-1
    if seed node is 1, there are one possible path: 1->0->-1->-1
    */
    // check the edge exists in corresponding homograph
    for (int j = 0; j < seeds_vector.size(); j++)
    {
        for (int i = 0; i < walk_length; i++)
        {
            torch::Tensor clonedTensor = actual_path_tensor.index({"...", j}).clone();
            int64_t *row = clonedTensor.unsqueeze(0).data_ptr<int64_t>();
            // torch::Tensor indices_CPU = A.GetCSC()->indices.to(torch::kCPU);
            // torch::Tensor indptr_CPU = A.GetCSC()->indptr.to(torch::kCPU);
            bool result = check_if_edge_exist(row[i], row[i + 1], indptr1, indices1);
            EXPECT_TRUE(result);
        }
    }
}