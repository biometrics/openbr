#include <iostream>
using namespace std;

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

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

    // copy the eigenvectors to the GPU
    cudaMalloc(&cudaEvPtr, evRows*evCols*sizeof(float));
    cudaMemcpy(cudaEvPtr, evPtr, evRows*evCols*sizeof(float), cudaMemcpyHostToDevice);

    // copy the mean to the GPU
    cudaMalloc(&cudaMeanPtr, meanElems*sizeof(float));
    cudaMemcpy(cudaMeanPtr, meanPtr, meanElems*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&_cudaSrcPtr, _meanElems*sizeof(float));
    cudaMalloc(&_cudaDstPtr, _evCols*sizeof(float));
  }

  void cudapca_trainwrapper(const void* cudaDataPtr, float* dataPtr, int rows, int cols) {
    cudaMemcpy(dataPtr, cudaDataPtr, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);
  }

  void cudapca_projectwrapper(void* src, void** dst) {
    // copy the image to the GPU
    //cudaMemcpy(_cudaSrcPtr, src, _meanElems*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(dst, _evRows*_evCols*sizeof(float));

    // subtract out the mean of the image (mean is 1xpixels in size)
    int threadsPerBlock = 64;
    int numBlocks = _meanElems / threadsPerBlock + 1;
    cudapca_project_subtractmean_kernel<<<numBlocks, threadsPerBlock>>>((float*)src, cudaMeanPtr, _meanElems);

    // perform the multiplication
    threadsPerBlock = 64;
    numBlocks = _evCols / threadsPerBlock + 1;
    cudapca_project_multiply_kernel<<<numBlocks, threadsPerBlock>>>((float*)src, (float*)(*dst), cudaEvPtr, _evRows, _evCols);

    //cudaFree(src);    // TODO(colin): figure out why adding this free causes memory corruption...

    // copy the data back to the CPU
    //cudaMemcpy(dst, _cudaDstPtr, _evCols*sizeof(float), cudaMemcpyDeviceToHost);
  }
}}
