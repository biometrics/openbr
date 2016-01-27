#include <opencv2/gpu/gpu.hpp>
#include <stdio.h>

using namespace cv;
using namespace cv::gpu;

#include "cudalbp.hpp"

namespace br { namespace cuda {
  __device__ __forceinline__ uint8_t cudalbp_kernel_get_pixel_value(int row, int col, uint8_t* srcPtr, size_t srcStep, int rows, int cols) {
    return (row >= rows || col >= cols) ? 0 : (srcPtr + row*srcStep)[col];
  }

  __global__ void cudalbp_kernel(uint8_t* srcPtr, uint8_t* dstPtr, size_t srcStep, size_t dstStep, int rows, int cols, uint8_t* lut)
  {
    int rowInd = blockIdx.y*blockDim.y+threadIdx.y;
    int colInd = blockIdx.x*blockDim.x+threadIdx.x;
    int radius = 1;

    // don't do anything if the index is out of bounds
    if (rowInd >= rows || colInd >= cols)
      return;

    const uint8_t cval = cudalbp_kernel_get_pixel_value(rowInd+0*radius, colInd+0*radius, srcPtr, srcStep, rows, cols);//(srcPtr[(rowInd*srcStep+0*radius)*m.cols+colInd+0*radius]);                      // center value
    uint8_t val = lut[(cudalbp_kernel_get_pixel_value(rowInd-1*radius, colInd-1*radius, srcPtr, srcStep, rows, cols) >= cval ? 128 : 0) |
                      (cudalbp_kernel_get_pixel_value(rowInd-1*radius, colInd+0*radius, srcPtr, srcStep, rows, cols) >= cval ? 64  : 0) |
                      (cudalbp_kernel_get_pixel_value(rowInd-1*radius, colInd+1*radius, srcPtr, srcStep, rows, cols) >= cval ? 32  : 0) |
                      (cudalbp_kernel_get_pixel_value(rowInd+0*radius, colInd+1*radius, srcPtr, srcStep, rows, cols) >= cval ? 16  : 0) |
                      (cudalbp_kernel_get_pixel_value(rowInd+1*radius, colInd+1*radius, srcPtr, srcStep, rows, cols) >= cval ? 8   : 0) |
                      (cudalbp_kernel_get_pixel_value(rowInd+1*radius, colInd+0*radius, srcPtr, srcStep, rows, cols) >= cval ? 4   : 0) |
                      (cudalbp_kernel_get_pixel_value(rowInd+1*radius, colInd-1*radius, srcPtr, srcStep, rows, cols) >= cval ? 2   : 0) |
                      (cudalbp_kernel_get_pixel_value(rowInd+0*radius, colInd-1*radius, srcPtr, srcStep, rows, cols) >= cval ? 1   : 0)];

    // store calculated value away in the right place
    uint8_t* dstRowPtr = dstPtr + rowInd*dstStep;
    dstRowPtr[colInd] = val;
  }

  void cudalbp_wrapper(GpuMat& src, GpuMat& dst, uint8_t* lut)
  {
    // convert the GpuMats to pointers
    uint8_t* srcPtr = (uint8_t*)src.data;
    uint8_t* dstPtr = (uint8_t*)dst.data;

    int imageWidth = src.cols;
    int imageHeight = src.rows;

    // make 8 * 8 = 64 square block
    dim3 threadsPerBlock(8, 8);

    dim3 numBlocks(imageWidth/threadsPerBlock.x + 1,
                   imageHeight/threadsPerBlock.y + 1);

    //printf("Src Image Dimesions:\n\trows: %d\tcols: %d\n", src.rows, src.cols);
    //printf("Dst Image Dimesions:\n\trows: %d\tcols: %d\n", dst.rows, dst.cols);
    //printf("Running CUDALBP\nBlock Dimensions:\n\tx: %d\ty: %d\n", numBlocks.x, numBlocks.y);

    cudalbp_kernel<<<numBlocks, threadsPerBlock>>>(srcPtr, dstPtr, src.step, dst.step, imageHeight, imageWidth, lut);
  }

  void cudalbp_init_wrapper(uint8_t* lut, uint8_t** lutGpuPtrPtr) {
    cudaMalloc(lutGpuPtrPtr, 256*sizeof(uint8_t));
    cudaMemcpy(*lutGpuPtrPtr, lut, 256*sizeof(uint8_t), cudaMemcpyHostToDevice);
  }
}}
