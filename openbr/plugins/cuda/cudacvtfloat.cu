/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2016 Li Li, Colin Heinzmann                                     *
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
    dim3 numBlocks(
      cols / threadsPerBlock.x + 1,
      rows / threadsPerBlock.y + 1
    );

    kernel<<<numBlocks, threadsPerBlock>>>((const unsigned char*)src, (float*)(*dst), rows, cols);
    CUDA_KERNEL_ERR_CHK(&err);

    // free the src memory since it is now in a newly allocated dst
    CUDA_SAFE_FREE(src, &err);
  }

}}}
