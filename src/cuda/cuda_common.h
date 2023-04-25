#ifndef GS_CUDA_CUDA_COMMON_H_
#define GS_CUDA_CUDA_COMMON_H_

#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/remove.h>
#include <torch/torch.h>
#include <cub/cub.cuh>

#define WARP_SIZE 32
#define MIN(x, y) ((x < y) ? x : y)

#define CUDA_CALL(func)                                      \
  {                                                          \
    cudaError_t e = (func);                                  \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) \
        << "CUDA: " << cudaGetErrorString(e);                \
  }

#define CUDA_KERNEL_CALL(kernel, nblks, nthrs, ...)               \
  {                                                               \
    (kernel)<<<(nblks), (nthrs)>>>(__VA_ARGS__);                  \
    cudaError_t e = cudaGetLastError();                           \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)      \
        << "CUDA kernel launch error: " << cudaGetErrorString(e); \
  }

#endif