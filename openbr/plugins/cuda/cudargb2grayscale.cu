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

  __global__ void cudargb2grayscale_kernel(uint8_t* srcPtr, uint8_t* dstPtr, int rows, int cols)
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

  void cudargb2grayscale_wrapper(void* srcPtr, void** dstPtr, int rows, int cols)
  {
    cudaError_t err;
    dim3 threadsPerBlock(9, 9);
    dim3 numBlocks(cols/threadsPerBlock.x + 1,
                   rows/threadsPerBlock.y + 1);
    CUDA_SAFE_MALLOC(dstPtr, rows*cols*sizeof(uint8_t), &err);

    cudargb2grayscale_kernel<<<numBlocks, threadsPerBlock>>>((uint8_t*)srcPtr, (uint8_t*) (*dstPtr), rows, cols);
    CUDA_KERNEL_ERR_CHK(&err);
    CUDA_SAFE_FREE(srcPtr, &err);
  } 

}}
