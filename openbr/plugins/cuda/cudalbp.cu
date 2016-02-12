#include <opencv2/gpu/gpu.hpp>
#include <stdio.h>

using namespace cv;
using namespace cv::gpu;

namespace br { namespace cuda {
  uint8_t* lut;

  __device__ __forceinline__ uint8_t cudalbp_kernel_get_pixel_value(int row, int col, uint8_t* srcPtr, int rows, int cols) {
    return (row >= rows || col >= cols) ? 0 : (srcPtr + row*cols)[col];
  }

  __global__ void cudalbp_kernel(uint8_t* srcPtr, uint8_t* dstPtr, int rows, int cols, uint8_t* lut)
  {
    int rowInd = blockIdx.y*blockDim.y+threadIdx.y;
    int colInd = blockIdx.x*blockDim.x+threadIdx.x;
    int radius = 1;

    // don't do anything if the index is out of bounds
    if (rowInd >= rows || colInd >= cols) {
      return;
    }

    const uint8_t cval = cudalbp_kernel_get_pixel_value(rowInd+0*radius, colInd+0*radius, srcPtr, rows, cols);//(srcPtr[(rowInd*srcStep+0*radius)*m.cols+colInd+0*radius]);                      // center value
    uint8_t val = lut[(cudalbp_kernel_get_pixel_value(rowInd-1*radius, colInd-1*radius, srcPtr, rows, cols) >= cval ? 128 : 0) |
                      (cudalbp_kernel_get_pixel_value(rowInd-1*radius, colInd+0*radius, srcPtr, rows, cols) >= cval ? 64  : 0) |
                      (cudalbp_kernel_get_pixel_value(rowInd-1*radius, colInd+1*radius, srcPtr, rows, cols) >= cval ? 32  : 0) |
                      (cudalbp_kernel_get_pixel_value(rowInd+0*radius, colInd+1*radius, srcPtr, rows, cols) >= cval ? 16  : 0) |
                      (cudalbp_kernel_get_pixel_value(rowInd+1*radius, colInd+1*radius, srcPtr, rows, cols) >= cval ? 8   : 0) |
                      (cudalbp_kernel_get_pixel_value(rowInd+1*radius, colInd+0*radius, srcPtr, rows, cols) >= cval ? 4   : 0) |
                      (cudalbp_kernel_get_pixel_value(rowInd+1*radius, colInd-1*radius, srcPtr, rows, cols) >= cval ? 2   : 0) |
                      (cudalbp_kernel_get_pixel_value(rowInd+0*radius, colInd-1*radius, srcPtr, rows, cols) >= cval ? 1   : 0)];

    // store calculated value away in the right place
    int index = rowInd*cols + colInd;
    dstPtr[index] = val;
  }

  //void cudalbp_wrapper(uint8_t* srcPtr, uint8_t* dstPtr, uint8_t* lut, int imageWidth, int imageHeight, size_t step)
  void cudalbp_wrapper(void* srcPtr, void** dstPtr, int rows, int cols)
  {
    // make 8 * 8 = 64 square block
    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks(cols/threadsPerBlock.x + 1,
                   rows/threadsPerBlock.y + 1);

    cudaMalloc(dstPtr, rows*cols*sizeof(uint8_t));
    cudalbp_kernel<<<numBlocks, threadsPerBlock>>>((uint8_t*)srcPtr, (uint8_t*)(*dstPtr), rows, cols, lut);
  }

  void cudalbp_init_wrapper(uint8_t* cpuLut) {
    cudaMalloc(&lut, 256*sizeof(uint8_t));
    cudaMemcpy(lut, cpuLut, 256*sizeof(uint8_t), cudaMemcpyHostToDevice);
  }
}}
