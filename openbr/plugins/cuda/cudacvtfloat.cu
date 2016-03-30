#include <iostream>
using namespace std;

#include "cudadefines.hpp"

namespace br { namespace cuda { namespace cvtfloat {

  __global__ void kernel(const unsigned char* src, float* dst, int rows, int cols) {
    // get my index
    int rowInd = blockIdx.y*blockDim.y + threadIdx.y;
    int colInd = blockIdx.x*blockDim.x + threadIdx.x;

    // bounds check
    if (rowInd >= rows || colInd >= cols) {
      return;
    }

    int index = rowInd*cols + colInd;
    dst[index] = (float)src[index];
  }

  void wrapper(void* src, void** dst, int rows, int cols) {
    cudaError_t err;
    CUDA_SAFE_MALLOC(dst, rows*cols*sizeof(float), &err);

    dim3 threadsPerBlock(8, 8);
    dim3 blocks(
      cols / threadsPerBlock.x + 1,
      rows / threadsPerBlock.y + 1
    );

    kernel<<<threadsPerBlock, blocks>>>((const unsigned char*)src, (float*)(*dst), rows, cols);
    CUDA_KERNEL_ERR_CHK(&err);

    // free the src memory since it is now in a newly allocated dst
    CUDA_SAFE_FREE(src, &err);
  }

}}}
