#include <math.h>

#include "cudadefines.hpp"

namespace br { namespace cuda { namespace L2 {

  __global__ void my_subtract_kernel(float* aPtr, float* bPtr, float* workPtr, int length) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;

    if (index >= length) {
      return;
    }

    // perform the subtraction in-place
    // use b because it is the comparison
    // image
    workPtr[index] = aPtr[index] - bPtr[index];
    workPtr[index] = workPtr[index] * workPtr[index];
  }

  __global__ void collapse_kernel(float* inPtr, float* outPtr, int length) {
    // make sure there is only one thread that we are calling
    if (blockIdx.x != 0 || threadIdx.x != 0) {
      return;
    }

    // sum up all the values
    *outPtr = 0;
    for (int i=0; i < length; i++) {
      *outPtr = *outPtr + inPtr[i];
    }

    // take the square root
    *outPtr = sqrtf(*outPtr);
  }

  void wrapper(float* cudaAPtr, float* cudaBPtr, int length, float* outPtr) {
    cudaError_t err;
    float* cudaOutPtr;
    CUDA_SAFE_MALLOC(&cudaOutPtr, sizeof(float), &err);

    float* cudaWorkBufferPtr;
    CUDA_SAFE_MALLOC(&cudaWorkBufferPtr, sizeof(float)*length, &err);

    // perform the subtraction
    int threadsPerBlock = 64;
    int numBlocks = length / threadsPerBlock + 1;
    my_subtract_kernel<<<threadsPerBlock, numBlocks>>>(cudaAPtr, cudaBPtr, cudaWorkBufferPtr, length);
    CUDA_KERNEL_ERR_CHK(&err);

    // perform the collapse
    collapse_kernel<<<1,1>>>(cudaWorkBufferPtr, cudaOutPtr, length);
    CUDA_KERNEL_ERR_CHK(&err);

    // copy the single value back to the destinsion
    CUDA_SAFE_MEMCPY(outPtr, cudaOutPtr, sizeof(float), cudaMemcpyDeviceToHost, &err);

    CUDA_SAFE_FREE(cudaOutPtr, &err);

    // do not free aPtr which should be the reference library
    // only free bPtr, which is the image we are comparing
    CUDA_SAFE_FREE(cudaBPtr, &err);
    CUDA_SAFE_FREE(cudaWorkBufferPtr, &err);
  }
}}}

// 128CUDAEigenfaces on 6400 ATT: 54.367s
// 128CUDAEigenfacesL2 on 6400 ATT: 
