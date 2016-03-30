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
  __global__ void multiplyKernel(float* src, float* intermediaryBuffer, float* evPtr, int evRows, int evCols, int stepSize) {
    int colInd = blockIdx.x*blockDim.x+threadIdx.x;
    if (colInd >= evCols) {
      return;
    }

    int stepNum = threadIdx.y;
    int iStart = stepNum*stepSize;
    int iEnd = iStart+stepSize;
    if (iStart >= evRows) {
      return;
    }
    if (iEnd > evRows) {
      iEnd = evRows;
    }

    float acc = 0;
    for (int i=iStart; i < iEnd; i++) {
      acc += evPtr[evCols*i + colInd] * src[i];
    }

    intermediaryBuffer[stepSize*stepNum + colInd] = acc;
  }

  __global__ void multiplyJoinKernel(float* intermediaryBuffer, float* dst, int evRows, int evCols, int numSteps, int stepSize) {
    int colInd = blockIdx.x*blockDim.x+threadIdx.x;
    if (colInd >= evCols) {
      return;
    }

    float acc = 0;
    for (int i = 0; i < numSteps; i++) {
      acc += intermediaryBuffer[stepSize*i + colInd];
    }

    dst[colInd] = acc;
  }

  __global__ void subtractMeanKernel(float* out, float* mean, int numCols) {
    int colInd = blockIdx.x*blockDim.x+threadIdx.x;

    // perform bound checking
    if (colInd >= numCols) {
      return;
    }

    // subtract out the mean
    out[colInd] -= mean[colInd];
  }

  float* cudaEvPtr; int _evRows; int _evCols;
  float* cudaMeanPtr; int _meanElems;
  float* _cudaSrcPtr;
  float* _cudaDstPtr;

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

    CUDA_SAFE_MALLOC(&_cudaSrcPtr, _meanElems*sizeof(float), &err);
    CUDA_SAFE_MALLOC(&_cudaDstPtr, _evCols*sizeof(float), &err);

    // initialize the intermediary working space,
    _numSteps = 16;
    _stepSize = _evRows / _numSteps + 1;
    CUDA_SAFE_MALLOC(&intermediaryBuffer, _numSteps*_evCols*sizeof(float), &err);
  }

  void wrapper(void* src, void** dst) {
    cudaError_t err;
    CUDA_SAFE_MALLOC(dst, _evCols*sizeof(float), &err);

    // subtract out the mean of the image (mean is 1xpixels in size)
    int threadsPerBlock = 64;
    int numBlocks = _meanElems / threadsPerBlock + 1;
    subtractMeanKernel<<<numBlocks, threadsPerBlock>>>((float*)src, cudaMeanPtr, _meanElems);
    CUDA_KERNEL_ERR_CHK(&err);

    // perform the multiplication
    dim3 threadsPerBlock2d(64, _numSteps);
    dim3 numBlocks2d(_evCols / threadsPerBlock2d.x + 1, 1);
    multiplyKernel<<<numBlocks2d, threadsPerBlock2d>>>((float*)src, intermediaryBuffer, cudaEvPtr, _evRows, _evCols, _stepSize);
    CUDA_KERNEL_ERR_CHK(&err);

    threadsPerBlock = 64;
    numBlocks = _evCols / threadsPerBlock + 1;
    multiplyJoinKernel<<<numBlocks, threadsPerBlock>>>(intermediaryBuffer, (float*)(*dst), _evRows, _evCols, _numSteps, _stepSize);
    CUDA_KERNEL_ERR_CHK(&err);

    CUDA_SAFE_FREE(src, &err);    // TODO(colin): figure out why adding this free causes memory corruption...
  }
}}}
