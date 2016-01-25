// note: Using 8-bit unsigned 1 channel images

#include <opencv2/gpu/gpu.hpp>

using namespace cv;
using namespace cv::gpu;

#include "passthrough.hpp"

namespace br { namespace cuda {
  __global__ void passthrough_kernel(uint8_t* srcPtr, uint8_t* dstPtr, size_t srcStep, size_t dstStep, int cols, int rows) {
    int rowInd = blockIdx.y*blockDim.y+threadIdx.y;
    int colInd = blockIdx.x*blockDim.x+threadIdx.x;

    // don't do anything if we are outside the allowable positions
    if (rowInd >= rows || colInd >= cols)
      return;

    uint8_t srcVal = (srcPtr + rowInd*srcStep)[colInd];
    uint8_t* rowDstPtr = dstPtr + rowInd*dstStep;

    rowDstPtr[colInd] = srcVal;
  }

  void passthrough_wrapper(GpuMat& src, GpuMat& dst) {
    // convert the GpuMats to pointers
    uint8_t* srcPtr = (uint8_t*)src.data;
    uint8_t* dstPtr = (uint8_t*)dst.data;

    int imageWidth = src.cols;
    int imageHeight = src.rows;

    // make 8 * 8 = 64 square block
    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks(imageWidth / threadsPerBlock.x + 1,
                   imageHeight / threadsPerBlock.y + 1);

    passthrough_kernel<<<numBlocks, threadsPerBlock>>>(srcPtr, dstPtr, src.step, dst.step, imageWidth, imageHeight);
  }
}}


// read http://stackoverflow.com/questions/31927297/array-of-ptrstepszgpumat-to-a-c-cuda-kernel
