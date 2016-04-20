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

#include <iostream>
using namespace std;

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "cudadefines.hpp"

using namespace cv;
using namespace cv::gpu;

/*
 * These are the CUDA functions for CUDAPCA.  See cudapca.cpp for more details
 */

namespace br { namespace cuda { namespace pca {
  __global__ void multiplyKernel(float* src, float* intermediaryBuffer, float* evPtr, int numEigenvectors, int numSteps, int stepSize, int numPixels) {
    int evIdx = blockIdx.x*blockDim.x+threadIdx.x;
    int stepIdx = blockIdx.y*blockDim.y+threadIdx.y;

    if (evIdx >= numEigenvectors || stepIdx >= numSteps) {
      return;
    }

    float acc = 0;
    int startIdx = stepSize*stepIdx;
    int stopIdx = startIdx+stepSize;
    if (startIdx >= numPixels) {
      return;
    }
    if (stopIdx >= numPixels) {
      stopIdx = numPixels;
    }
    for(int i=startIdx; i < stopIdx; i++) {
      acc += src[i]*evPtr[i*numEigenvectors + evIdx];
    }

    intermediaryBuffer[stepIdx*stepSize + evIdx] = acc;
  }

  __global__ void multiplyJoinKernel(float* intermediaryBuffer, float* out, int numEigenvectors, int numSteps, int stepSize) {
    int evIdx = blockIdx.x*blockDim.x+threadIdx.x;
    if (evIdx >= numEigenvectors) {
      return;
    }

    float acc = 0;
    for (int i=0; i < numSteps; i++) {
      int ibIdx = i*stepSize + evIdx;
      if (ibIdx >= numSteps*stepSize) {
        break;
      }
      acc += intermediaryBuffer[ibIdx];
    }

    out[evIdx] = acc;
  }

  __global__ void subtractMeanKernel(float* out, float* mean, int numElems) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    // perform bound checking
    if (idx >= numElems) {
      return;
    }

    // subtract out the mean
    out[idx] -= mean[idx];
  }

  // _evRows: the number of pixels in the trained images
  // _evCols: the number of eigenvectors
  // _meanElems: the number of pixels in an image
  // _stepSize: the number of pixels in a single step
  // _numSteps: the number of steps required to complete operation
  float* cudaEvPtr; int _evRows; int _evCols;
  float* cudaMeanPtr; int _meanElems;
  int _numSteps; int _stepSize;
  float* intermediaryBuffer;

  void initializeWrapper(float* evPtr, int evRows, int evCols, float* meanPtr, int meanElems) {
    _evRows = evRows; _evCols = evCols;
    _meanElems = meanElems;

    cudaError_t err;

    // copy the eigenvectors to the GPU
    CUDA_SAFE_MALLOC(&cudaEvPtr, evRows*evCols*sizeof(float), &err);
    CUDA_SAFE_MEMCPY(cudaEvPtr, evPtr, evRows*evCols*sizeof(float), cudaMemcpyHostToDevice, &err);

    // copy the mean to the GPU
    CUDA_SAFE_MALLOC(&cudaMeanPtr, meanElems*sizeof(float), &err);
    CUDA_SAFE_MEMCPY(cudaMeanPtr, meanPtr, meanElems*sizeof(float), cudaMemcpyHostToDevice, &err);

    // initialize the intermediary working space,
    _stepSize = 2048;
    _numSteps = _evRows / _stepSize + 1;
    CUDA_SAFE_MALLOC(&intermediaryBuffer, _numSteps*_stepSize*sizeof(float), &err);
  }

  void trainWrapper(void* cudaSrc, float* data, int rows, int cols) {
    cudaError_t err;
    CUDA_SAFE_MEMCPY(data, cudaSrc, rows*cols*sizeof(float), cudaMemcpyDeviceToHost, &err);
  }

  void wrapper(void* src, void** dst, int imgRows, int imgCols) {
    cudaError_t err;
    CUDA_SAFE_MALLOC(dst, _evCols*sizeof(float), &err);

    if (imgRows*imgCols != _evRows || imgRows*imgCols != _meanElems) {
      cout << "ERR: Image dimension mismatch!" << endl;
      throw 0;
    }

    // subtract out the mean of the image (mean is 1xpixels in size), perform in place (in src)
    int threadsPerBlock = 512;
    int numBlocks = _meanElems / threadsPerBlock + 1;
    subtractMeanKernel<<<numBlocks, threadsPerBlock>>>((float*)src, cudaMeanPtr, _meanElems);
    CUDA_KERNEL_ERR_CHK(&err);

    // perform matrix multiplication
    dim3 threadsPerBlock2d(512, 1);
    dim3 numBlocks2d(
        _evCols / threadsPerBlock2d.x + 1,
        _numSteps / threadsPerBlock2d.y + 1);
    multiplyKernel<<<numBlocks2d, threadsPerBlock2d>>>((float*)src, intermediaryBuffer, cudaEvPtr, _evCols, _numSteps, _stepSize, _meanElems);
    CUDA_KERNEL_ERR_CHK(&err);

    threadsPerBlock = 512;
    numBlocks = _evCols / threadsPerBlock + 1;
    multiplyJoinKernel<<<numBlocks, threadsPerBlock>>>(intermediaryBuffer, (float*)*dst, _evCols, _numSteps, _stepSize);
    CUDA_KERNEL_ERR_CHK(&err);

    // free the src memory
    CUDA_SAFE_FREE(src, &err);
  }
}}}
