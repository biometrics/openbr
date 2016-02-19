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

namespace br{ namespace cuda {

  __global__ void cudacvt_kernel(uint8_t* srcPtr, uint8_t* dstPtr, int rows, int cols)
  {
    int rowInd = blockIdx.y*blockDim.y+threadIdx.y;
    int colInd = blockIdx.x*blockDim.x+threadIdx.x;
    int index = rowInd*cols + colInd;
    if (rowInd < 1 || rowInd >= rows-1 || colInd < 1 || colInd >= cols-1) {
      if (rowInd >= rows || colInd >= cols) {
        return;
      } else {
        return;
      }
    }

    dstPtr[index] = 0.299f * srcPtr[3*index] + 0.587f * srcPtr[3*index+1] + 0.114f * srcPtr[3*index+2];
    return;
  }

  void cudacvt_wrapper(void* srcPtr, void** dstPtr, int rows, int cols)
  {
    cudaError_t err;
    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks(cols/threadsPerBlock.x + 1,
                   rows/threadsPerBlock.y + 1);
    std::cout << "Before malloc" << std::endl;
    CUDA_SAFE_MALLOC(dstPtr, rows*cols*sizeof(uint8_t), &err);
    std::cout << "After malloc" << std::endl;

    cudacvt_kernel<<<numBlocks, threadsPerBlock>>>((uint8_t*)srcPtr, (uint8_t*) (*dstPtr), rows, cols);
    CUDA_KERNEL_ERR_CHK(&err);
    CUDA_SAFE_FREE(srcPtr, &err);
  } 

}}
