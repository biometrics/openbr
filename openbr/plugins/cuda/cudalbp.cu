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

#include <opencv2/gpu/gpu.hpp>
#include <stdio.h>

#include "cudadefines.hpp"

using namespace cv;
using namespace cv::gpu;

/*
 * These are the CUDA functions for CUDALBP.  See cudapca.cpp for more details
 */

namespace br { namespace cuda { namespace lbp {
  uint8_t* lut;

  __device__ __forceinline__ uint8_t getPixelValueKernel(int row, int col, uint8_t* srcPtr, int rows, int cols) {
    return (srcPtr + row*cols)[col];
  }

  __global__ void lutKernel(uint8_t* srcPtr, uint8_t* dstPtr, int rows, int cols, uint8_t* lut)
  {
    int rowInd = blockIdx.y*blockDim.y+threadIdx.y;
    int colInd = blockIdx.x*blockDim.x+threadIdx.x;
    int radius = 1;

    int index = rowInd*cols + colInd;

    // don't do anything if the index is out of bounds
    if (rowInd < 1 || rowInd >= rows-1 || colInd < 1 || colInd >= cols-1) {
      if (rowInd >= rows || colInd >= cols) {
        return;
      } else {
        dstPtr[index] = 0;
        return;
      }
    }

    const uint8_t cval = getPixelValueKernel(rowInd+0*radius, colInd+0*radius, srcPtr, rows, cols);//(srcPtr[(rowInd*srcStep+0*radius)*m.cols+colInd+0*radius]);                      // center value
    uint8_t val = lut[(getPixelValueKernel(rowInd-1*radius, colInd-1*radius, srcPtr, rows, cols) >= cval ? 128 : 0) |
                      (getPixelValueKernel(rowInd-1*radius, colInd+0*radius, srcPtr, rows, cols) >= cval ? 64  : 0) |
                      (getPixelValueKernel(rowInd-1*radius, colInd+1*radius, srcPtr, rows, cols) >= cval ? 32  : 0) |
                      (getPixelValueKernel(rowInd+0*radius, colInd+1*radius, srcPtr, rows, cols) >= cval ? 16  : 0) |
                      (getPixelValueKernel(rowInd+1*radius, colInd+1*radius, srcPtr, rows, cols) >= cval ? 8   : 0) |
                      (getPixelValueKernel(rowInd+1*radius, colInd+0*radius, srcPtr, rows, cols) >= cval ? 4   : 0) |
                      (getPixelValueKernel(rowInd+1*radius, colInd-1*radius, srcPtr, rows, cols) >= cval ? 2   : 0) |
                      (getPixelValueKernel(rowInd+0*radius, colInd-1*radius, srcPtr, rows, cols) >= cval ? 1   : 0)];

    // store calculated value away in the right place
    dstPtr[index] = val;
  }

  //void cudalbp_wrapper(uint8_t* srcPtr, uint8_t* dstPtr, uint8_t* lut, int imageWidth, int imageHeight, size_t step)
  void wrapper(void* srcPtr, void** dstPtr, int rows, int cols)
  {
    cudaError_t err;

    // make 8 * 8 = 64 square block
    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks(cols/threadsPerBlock.x + 1,
                   rows/threadsPerBlock.y + 1);

    CUDA_SAFE_MALLOC(dstPtr, rows*cols*sizeof(uint8_t), &err);
    lutKernel<<<numBlocks, threadsPerBlock>>>((uint8_t*)srcPtr, (uint8_t*)(*dstPtr), rows, cols, lut);
    CUDA_KERNEL_ERR_CHK(&err);

    CUDA_SAFE_FREE(srcPtr, &err);
  }

  void initializeWrapper(uint8_t* cpuLut) {
    cudaError_t err;
    CUDA_SAFE_MALLOC(&lut, 256*sizeof(uint8_t), &err);
    CUDA_SAFE_MEMCPY(lut, cpuLut, 256*sizeof(uint8_t), cudaMemcpyHostToDevice, &err);
  }
}}}
