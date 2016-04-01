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

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "cudadefines.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>

using namespace std;
using namespace cv;
using namespace cv::gpu;

namespace br { namespace cuda { namespace rgb2grayscale {

  __global__ void kernel(uint8_t* srcPtr, uint8_t* dstPtr, int rows, int cols)
  {
    int rowInd = blockIdx.y*blockDim.y+threadIdx.y;
    int colInd = blockIdx.x*blockDim.x+threadIdx.x;
    int index = rowInd*cols + colInd;
    if (rowInd < 0 || rowInd >= rows || colInd < 0 || colInd >= cols) {
      return;
    }
    int new_index = 3 * index;
    float g = (float) srcPtr[new_index];
    float b = (float) srcPtr[new_index+1];
    float r = (float) srcPtr[new_index+2];

    dstPtr[index] = (uint8_t) (0.299f * g + 0.587f * b + 0.114f * r);
    return;
  }

  void wrapper(void* srcPtr, void** dstPtr, int rows, int cols)
  {
    cudaError_t err;
    dim3 threadsPerBlock(9, 9);
    dim3 numBlocks(cols/threadsPerBlock.x + 1,
                   rows/threadsPerBlock.y + 1);
    CUDA_SAFE_MALLOC(dstPtr, rows*cols*sizeof(uint8_t), &err);

    kernel<<<numBlocks, threadsPerBlock>>>((uint8_t*)srcPtr, (uint8_t*) (*dstPtr), rows, cols);
    CUDA_KERNEL_ERR_CHK(&err);
    CUDA_SAFE_FREE(srcPtr, &err);
  }

}}}
