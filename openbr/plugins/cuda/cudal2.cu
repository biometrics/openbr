/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2016 Colin Heinzmann                                            *
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License");           *
 * you may not use this file except in compliance with the License.          *
 * You may obtain a copy of the License at                                   *
 *                                                                           *
 *     http://www.apache.org/licenses/LICENSE-2.0                            *
 *                                                                           *
 * Unless required by applicable law or agreed to in writing, software       *
 * distributed under the License is distributed on an "AS IS" BASIS,         *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 * See the License for the specific language governing permissions and       *
 * limitations under the License.                                            *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include <math.h>


#include "cudadefines.hpp"

namespace br { namespace cuda { namespace L2 {

  __global__ void subtractKernel(float* aPtr, float* bPtr, float* workPtr, int length) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;

    if (index >= length) {
      return;
    }

    // perform the subtraction
    float res = aPtr[index] - bPtr[index];
    res = res * res;
    workPtr[index] = res;
  }

  __global__ void collapseKernel(float* inPtr, float* outPtr, int length) {
    // make sure there is only one thread that we are calling
    if (blockIdx.x != 0 || threadIdx.x != 0) {
      return;
    }

    // sum up all the values
    float acc = 0;
    for (int i=0; i < length; i++) {
      acc += inPtr[i];
    }

    *outPtr = acc;
  }

  float* cudaAPtr = NULL;
  float* cudaBPtr = NULL;
  float* cudaWorkBufferPtr = NULL;
  float* cudaOutPtr = NULL;
  int bufferLen = 0;

  void wrapper(float const* aPtr, float const* bPtr, int length, float* outPtr) {
    cudaError_t err;

    // allocate memory for the mats and copy data to graphics card
    // only allocate if there is a mismatch in image size, otherwise
    // use the existing allocated memory
    if (length != bufferLen) {
      if (cudaAPtr != NULL) {
        CUDA_SAFE_FREE(cudaAPtr, &err);
        CUDA_SAFE_FREE(cudaBPtr, &err);
        CUDA_SAFE_FREE(cudaWorkBufferPtr, &err);
        CUDA_SAFE_FREE(cudaOutPtr, &err);
      }
      CUDA_SAFE_MALLOC(&cudaAPtr, length*sizeof(float), &err);
      CUDA_SAFE_MALLOC(&cudaBPtr, length*sizeof(float), &err);
      CUDA_SAFE_MALLOC(&cudaWorkBufferPtr, sizeof(float)*length, &err);
      CUDA_SAFE_MALLOC(&cudaOutPtr, sizeof(float), &err);
      bufferLen = length;
    }

    // copy data over from CPU
    CUDA_SAFE_MEMCPY(cudaAPtr, aPtr, length*sizeof(float), cudaMemcpyHostToDevice, &err);
    CUDA_SAFE_MEMCPY(cudaBPtr, bPtr, length*sizeof(float), cudaMemcpyHostToDevice, &err);

    // perform the subtraction
    int threadsPerBlock = 512;
    int numBlocks = length / threadsPerBlock + 1;
    subtractKernel<<<threadsPerBlock, numBlocks>>>(cudaAPtr, cudaBPtr, cudaWorkBufferPtr, length);
    CUDA_KERNEL_ERR_CHK(&err);

    // perform the collapse
    collapseKernel<<<1,1>>>(cudaWorkBufferPtr, cudaOutPtr, length);
    CUDA_KERNEL_ERR_CHK(&err);

    // copy the single value back to the destinsion
    CUDA_SAFE_MEMCPY(outPtr, cudaOutPtr, sizeof(float), cudaMemcpyDeviceToHost, &err);
  }
}}}
