#include <iostream>
using namespace std;

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace cv;
using namespace cv::gpu;

#include "cudapca.hpp"

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

  __global__ void cudapca_project_subtractmean_kernel(float* out, float* mean, int cols) {
    int colInd = blockIdx.x*blockDim.x+threadIdx.x;

    // perform bound checking
    if (colInd >= cols) {
      return;
    }

    // subtract out the mean
    out[colInd] -= mean[colInd];
  }

  float* cudaEvPtr; int _evRows; int _evCols;
  float* cudaMeanPtr; int _meanElems;

  void cudapca_initwrapper() {

  }

  void cudapca_loadwrapper(float* evPtr, int evRows, int evCols, float* meanPtr, int meanElems) {
    _evRows = evRows; _evCols = evCols;
    _meanElems = meanElems;

    // copy the eigenvectors to the GPU
    cudaMalloc(&cudaEvPtr, evRows*evCols*sizeof(float));
    cudaMemcpy(cudaEvPtr, evPtr, evRows*evCols*sizeof(float), cudaMemcpyHostToDevice);

    // copy the mean to the GPU
    cudaMalloc(&cudaMeanPtr, meanElems*sizeof(float));
    cudaMemcpy(cudaMeanPtr, meanPtr, meanElems*sizeof(float), cudaMemcpyHostToDevice);
  }

  void cudapca_trainwrapper() {
    /*
    if (trainingSet[0].type() != CV_32FC1) {
      std::cout << "ERR: Requires single 32-bit floating point matrix!";
      return;
    }

    cudaError_t status;

    const int numExamples = trainingSetSize;
    int numPixels = trainingSet[0].rows * trainingSet[0].cols;

    // create a custom matrix
    float* cudaDataPtr;
    status = cudaMalloc(&cudaDataPtr, numPixels * numExamples * sizeof(float));
    if (status != cudaSuccess) {
      std::cout << "ERR: Memory allocation" << std::endl;
      return;
    }

    // copy all the data to the graphics card
    for (int i=0; i < numExamples; i++) {
      status = cudaMemcpy(cudaDataPtr + i*numPixels, trainingSet[i].ptr<float>(), numPixels*sizeof(float), cudaMemcpyHostToDevice);
      if (status != cudaSuccess) {
        std::cout << "ERR: Memcpy at index " << i << std::endl;
        return;
      }
    }

    // start the core part of the algorithm
    int numDimensions = numPixels;
    const bool dominantEigenEstimation = (numDimensions > numExamples);

    // malloc and init mean
    mean = new float[numDimensions];
    for (int i=0; i < numDimensions; i++) {
      mean[i] = 0;
    }
    float* cudaMeanPtr;
    status = cudaMalloc(&cudaMeanPtr, numDimensions*sizeof(float));
    if (status != cudaSuccess) {
      std::cout << " ERR: Malloc of mean" << std::endl;
      return;
    }

    if (keep != 0) {
      // compute the mean so we can subtract from data
      for (int i=0; i < numExamples; i++) {
        Mat& m = trainingSet[i];

        for (int j=0; j < numDimensions; j++) {
          mean[j] += m.ptr<float>()[i*numDimensions + j];
        }
      }
      for (int i=0; i < numDimensions; i++) {
        mean[i] = mean[i] / numExamples;
      }

      // copy mean over to graphics card
      cudaMemcpy(cudaMeanPtr, mean, numExamples*sizeof(float), cudaMemcpyHostToDevice);
      if (status != cudaSuccess) {
        std::cout << " ERR: Cpy of mean" << std::endl;
        return;
      }

      // set the thread dimensions and run the kernel
      dim3 threadsPerBlock(64, 1);
      dim3 numBlocks(numDimensions/threadsPerBlock.x + 1,
                     numExamples/threadsPerBlock.y + 1);

      subtractMean_kernel<<<numBlocks, threadsPerBlock>>>(cudaDataPtr, cudaMeanPtr, numExamples, numDimensions);

      // calculate the covariance matrix using kernel
      // malloc location for covariance matrix
      float* cudaCovPtr;
      status = cudaMalloc(&cudaCovPtr, numExamples*numExamples*sizeof(float));
      if (status != cudaSuccess) h
        std::cout << " ERR: Cpy of mean" << std::endl;
        return;
      }

      // calculate the covariance matrix
      threadsPerBlock = dim3(8, 8);
      numBlocks = dim3(numExamples/threadsPerBlock.x + 1,
                       numExamples/threadsPerBlock.y + 1);
      calculateCovariance_kernel<<<numBlocks, threadsPerBlock>>>(cudaDataPtr, cudaCovPtr, numExamples, numDimensions);

      // perform eigendecomposition
      //std::cout << "Skipping eigendecomposition" << std::endl;
      cusolverStatus_t cusolverStatus;
      cusolverStatus = cusolverDnSgebrd(cusolverHandle,)
    }
    */
  }

  void cudapca_projectwrapper(float* src, float* dst) {
    // copy the image to the GPU
    float* cudaSrcPtr;
    cudaMalloc(&cudaSrcPtr, _meanElems*sizeof(float));
    cudaMemcpy(cudaSrcPtr, src, _meanElems*sizeof(float), cudaMemcpyHostToDevice);

    float* cudaDstPtr;
    cudaMalloc(&cudaDstPtr, _evCols*sizeof(float));

    // subtract out the mean of the image (mean is 1xpixels in size)
    int threadsPerBlock = 64;
    int numBlocks = _meanElems / threadsPerBlock;
    cudapca_project_subtractmean_kernel<<<numBlocks, threadsPerBlock>>>(cudaSrcPtr, cudaMeanPtr, _meanElems);

    // perform the multiplication
    threadsPerBlock = 64;
    numBlocks = _evCols / threadsPerBlock;
    cudapca_project_multiply_kernel<<<numBlocks, threadsPerBlock>>>(cudaSrcPtr, cudaDstPtr, cudaEvPtr, _evRows, _evCols);

    // copy the data back to the CPU
    cudaMemcpy(dst, cudaDstPtr, _evCols*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(cudaSrcPtr);
    cudaFree(cudaDstPtr);
  }
}}
