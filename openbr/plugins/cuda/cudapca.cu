#include <iostream>
using namespace std;

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "cudadefines.hpp"

using namespace cv;
using namespace cv::gpu;

namespace br { namespace cuda {
  __global__ void calculateCovariance_kernel(float* trainingSet, float* cov, int numRows, int numCols) {
    int rowInd = blockIdx.y*blockDim.y + threadIdx.y;
    int colInd = blockIdx.x*blockDim.x + threadIdx.x;

    // this calculates trainingSet' * trainingSet
    if (rowInd >= numRows || colInd >= numCols) {
      return;
    }

    // get a reference the value we wish to write
    float& out = cov[rowInd*numRows + colInd];

    // calculate the value of this position
    out = 0;
    for (int i=0; i<numRows; i++) {
      out += trainingSet[rowInd*numCols + colInd] * trainingSet[rowInd*numCols + numRows]; // XXX(colin): not sure if this is correct
    }
    out = out / (numRows-1);
  }

  __global__ void cudapca_project_multiply_kernel(float* src, float* dst, float* evPtr, int evRows, int evCols) {
    int colInd = blockIdx.x*blockDim.x+threadIdx.x;

    // check dimensions
    if (colInd >= evCols) {
      return;
    }

    dst[colInd] = 0;
    for (int i=0; i < evRows; i++) {
      dst[colInd] += evPtr[evCols*i + colInd] * src[i];
    }
  }

  __global__ void cudapca_project_subtractmean_kernel(float* out, float* mean, int numCols) {
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

  void cudapca_loadwrapper(float* evPtr, int evRows, int evCols, float* meanPtr, int meanElems) {
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
  }

  void cudapca_trainwrapper(void* cudaDataPtr, float* dataPtr, int rows, int cols) {
    cudaError_t err;
    CUDA_SAFE_MEMCPY(dataPtr, cudaDataPtr, rows*cols*sizeof(float), cudaMemcpyDeviceToHost, &err);
    CUDA_SAFE_FREE(cudaDataPtr, &err);
  }

  void cudapca_projectwrapper(void* src, void** dst) {
    // copy the image to the GPU
    //cudaMemcpy(_cudaSrcPtr, src, _meanElems*sizeof(float), cudaMemcpyHostToDevice);
    cudaError_t err;
    CUDA_SAFE_MALLOC(dst, _evRows*_evCols*sizeof(float), &err);

    // subtract out the mean of the image (mean is 1xpixels in size)
    int threadsPerBlock = 64;
    int numBlocks = _meanElems / threadsPerBlock + 1;
    cudapca_project_subtractmean_kernel<<<numBlocks, threadsPerBlock>>>((float*)src, cudaMeanPtr, _meanElems);
    CUDA_KERNEL_ERR_CHK(&err);

    // perform the multiplication
    threadsPerBlock = 64;
    numBlocks = _evCols / threadsPerBlock + 1;
    cudapca_project_multiply_kernel<<<numBlocks, threadsPerBlock>>>((float*)src, (float*)(*dst), cudaEvPtr, _evRows, _evCols);
    CUDA_KERNEL_ERR_CHK(&err);

    CUDA_SAFE_FREE(src, &err);    // TODO(colin): figure out why adding this free causes memory corruption...

    // copy the data back to the CPU
    //cudaMemcpy(dst, _cudaDstPtr, _evCols*sizeof(float), cudaMemcpyDeviceToHost);
  }
}}
