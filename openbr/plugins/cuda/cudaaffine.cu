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

#include <opencv2/gpu/gpu.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <math.h>

#include "cudadefines.hpp"

using namespace cv;
using namespace cv::gpu;

namespace br { namespace cuda { namespace affine {

    __device__ __forceinline__ uint8_t getPixelValueDevice(int row, int col, uint8_t* srcPtr, int rows, int cols) {
        return (srcPtr + row*cols)[col];
    }


    __device__ __forceinline__ uint8_t getBilinearPixelValueDevice(double row, double col, uint8_t* srcPtr, int rows, int cols) {
        // http://www.sci.utah.edu/~acoste/uou/Image/project3/ArthurCOSTE_Project3.pdf
        // Bilinear Transformation
        // f(Px, Py) = f(Q11)×(1−Rx)×(1−Sy)+f(Q21)×(Rx)×(1−Sy)+f(Q12)×(1−Rx)×(Sy)+f(Q22)×(Rx)×(Sy)

        int row1 = floor(row);
        int row2 = row1+1;

        int col1 = floor(col);
        int col2 = col1+1;

        double d_row = row - row1;
        double d_col = col - col1;

        int Q11 = getPixelValueDevice(row1, col1, srcPtr, rows, cols);
        int Q21 = getPixelValueDevice(row2, col1, srcPtr, rows, cols);
        int Q12 = getPixelValueDevice(row1, col2, srcPtr, rows, cols);
        int Q22 = getPixelValueDevice(row2, col2, srcPtr, rows, cols);

        double val = Q22*(d_row*d_col) + Q12*((1-d_row)*d_col) + Q21*(d_row*(1-d_col)) + Q11*((1-d_row)*(1-d_col));
        return ((uint8_t) round(val));
    }

    __device__ __forceinline__ uint8_t getDistancePixelValueDevice(double row, double col, uint8_t* srcPtr, int rows, int cols) {
        int row1 = floor(row);
        int row2 = row1+1;

        int col1 = floor(col);
        int col2 = col1+1;

        double m1 = row2 - row;
        double m12 = m1*m1;

        double m2 = col - col1;
        double m22 = m2*m2;

        double d1 = sqrt(m12 - 2*m1 + 1 + m22);
        double d2 = sqrt(m12 + m22);
        double d3 = sqrt(m12 - 2*m1 + 1 + m22 - 2*m2 + 1);
        double d4 = sqrt(m12 + m22 - 2*m2 + 1);
        double sum = d1 + d2 + d3 + d4;

        double w1 = d1/sum;
        double w2 = d2/sum;
        double w3 = d3/sum;
        double w4 = d4/sum;

        uint8_t v1 = getPixelValueDevice(row1, col1, srcPtr, rows, cols);
        uint8_t v2 = getPixelValueDevice(row2, col1, srcPtr, rows, cols);
        uint8_t v3 = getPixelValueDevice(row1, col2, srcPtr, rows, cols);
        uint8_t v4 = getPixelValueDevice(row2, col2, srcPtr, rows, cols);

        return round(w1*v1 + w2*v2 + w3*v3 + w4*v4);
    }

    /*
     * trans_inv          - A pointer to a one-dimensional representation of the inverse of the transform matrix 3x3
     * dst_row            - The destination row (mapping to this row)
     * dst_col            - The destination column (mapping to this column)
     * src_row            - The computed source pixel row (mapping from this row)
     * src_col            - The computed source pixel column (mapping from this col)
     */
    __device__ __forceinline__ void getSrcCoordDevice(double *trans_inv, int dst_row, int dst_col, double* src_row_pnt, double* src_col_pnt){
        *src_col_pnt = dst_col * trans_inv[0] + dst_row * trans_inv[3] + trans_inv[6];
        *src_row_pnt = dst_col * trans_inv[1] + dst_row * trans_inv[4] + trans_inv[7];
		}

    __global__ void bilinearKernel(uint8_t* srcPtr, uint8_t* dstPtr, int srcRows, int srcCols, int dstRows, int dstCols) {
      int dstRowInd = blockIdx.y*blockDim.y+threadIdx.y;
      int dstColInd = blockIdx.x*blockDim.x+threadIdx.x;
      int dstIndex = dstRowInd*dstCols+dstColInd;

      // destination boundary checking
      if (dstRowInd >= dstRows || dstColInd >= dstCols) {
        return;
      }

      // get the reference indices and relative amounts
      float exactSrcRowInd = (float)dstRowInd / (float)dstRows * (float)srcRows;
      int minSrcRowInd = (int)exactSrcRowInd;
      int maxSrcRowInd = minSrcRowInd+1;
      float relSrcRowInd = 1.-(exactSrcRowInd-(float)minSrcRowInd);

      // get the reference indices and relative amounts
      double exactSrcColInd = (double)dstColInd / (double)dstCols * (double)srcCols;
      int minSrcColInd = (int)exactSrcColInd;
      int maxSrcColInd = minSrcColInd+1;
      float relSrcColInd = 1.-(exactSrcColInd-(float)minSrcColInd);

      // perform boundary checking
      if (minSrcRowInd < 0 || maxSrcRowInd >= srcRows || minSrcColInd < 0 || maxSrcColInd >= srcCols) {
        dstPtr[dstIndex] = 0;
        return;
      }

      // get each of the pixel values
      float topLeft = srcPtr[minSrcRowInd*srcCols+minSrcColInd];
      float topRight = srcPtr[minSrcRowInd*srcCols+maxSrcColInd];
      float bottomLeft = srcPtr[maxSrcRowInd*srcCols+minSrcColInd];
      float bottomRight = srcPtr[maxSrcRowInd*srcCols+maxSrcColInd];

      float out = relSrcRowInd*relSrcColInd*topLeft + relSrcRowInd*(1.-relSrcColInd)*topRight + (1.-relSrcRowInd)*relSrcColInd*bottomLeft + (1.-relSrcRowInd)*(1.-relSrcColInd)*bottomRight;

      dstPtr[dstIndex] = (int)out;
    }

    __global__ void affineKernel(uint8_t* srcPtr, uint8_t* dstPtr, double* trans_inv, int src_rows, int src_cols, int dst_rows, int dst_cols){
        int dstRowInd = blockIdx.y*blockDim.y+threadIdx.y;
        int dstColInd = blockIdx.x*blockDim.x+threadIdx.x;
        int dstIndex = dstRowInd*dst_cols + dstColInd;

        double srcRowPnt;
        double srcColPnt;

        // don't do anything if the index is out of bounds
        if (dstRowInd >= dst_rows || dstColInd >= dst_cols) {
          return;
        }
        if (dstRowInd == 0 || dstRowInd == dst_rows-1 || dstColInd ==0 || dstColInd == dst_cols-1) {
          dstPtr[dstIndex] = 0;
          return;
        }

        getSrcCoordDevice(trans_inv, dstRowInd, dstColInd, &srcRowPnt, &srcColPnt);
        const uint8_t cval = getBilinearPixelValueDevice(srcRowPnt, srcColPnt, srcPtr, src_rows, src_cols); // Get initial pixel value

        dstPtr[dstIndex] = cval;
    }

    void resizeWrapper(void* srcPtr, void** dstPtr, int srcRows, int srcCols, int dstRows, int dstCols) {
      // perform bilinear filtering

      // allocate space for destination
      cudaError_t err;
      CUDA_SAFE_MALLOC(dstPtr, dstRows*dstCols*sizeof(uint8_t), &err);

      // call the bilinear kernel function
      dim3 threadsPerBlock(32, 16);
      dim3 numBlocks(dstCols/threadsPerBlock.x + 1,
                     dstRows/threadsPerBlock.y + 1);

      bilinearKernel<<<numBlocks, threadsPerBlock>>>((uint8_t*)srcPtr, (uint8_t*)*dstPtr, srcRows, srcCols, dstRows, dstCols);
      CUDA_KERNEL_ERR_CHK(&err);

      CUDA_SAFE_FREE(srcPtr, &err);
    }

    void wrapper(void* srcPtr, void** dstPtr, Mat affineTransform, int src_rows, int src_cols, int dst_rows, int dst_cols) {
        cudaError_t err;
        double* gpuInverse;

        dim3 threadsPerBlock(32, 16);
        dim3 numBlocks(dst_cols/threadsPerBlock.x + 1,
                       dst_rows/threadsPerBlock.y + 1);

        //************************************************************************
        // Input affine is a 2x3 Mat whose transpose is used in the computations
        // [x, y, 1] = [u, v, 1] [ a^T | [0 0 1]^T ]
        // See "Digital Image Warping" by George Wolburg (p. 50)
        //************************************************************************

        // get new transform elements
        double a11 = affineTransform.at<double>(0, 0);
        double a12 = affineTransform.at<double>(1, 0);
        double a21 = affineTransform.at<double>(0, 1);
        double a22 = affineTransform.at<double>(1, 1);
        double a31 = affineTransform.at<double>(0, 2);
        double a32 = affineTransform.at<double>(1, 2);

        // compute transform inverse
        double det = 1 / (a11*a22 - a21*a12);

        double affineInverse[9];
        affineInverse[0] = a22 * det;
        affineInverse[1] = -a12 * det;
        affineInverse[2] = 0;
        affineInverse[3] = -a21 * det;
        affineInverse[4] = a11 * det;
        affineInverse[5] = 0;
        affineInverse[6] = (a21*a32 - a31*a22) * det;
        affineInverse[7] = (a31*a12 - a11*a32) * det;
        affineInverse[8] = (a11*a22 - a21*a12) * det;

        CUDA_SAFE_MALLOC(dstPtr, dst_rows*dst_cols*sizeof(uint8_t), &err);
        CUDA_SAFE_MALLOC(&gpuInverse, 3*3*sizeof(double), &err);

        CUDA_SAFE_MEMCPY(gpuInverse, affineInverse, 9*sizeof(double), cudaMemcpyHostToDevice, &err);

        affineKernel<<<numBlocks, threadsPerBlock>>>((uint8_t*)srcPtr, (uint8_t*)(*dstPtr), gpuInverse, src_rows, src_cols, dst_rows, dst_cols);
        CUDA_KERNEL_ERR_CHK(&err);

        CUDA_SAFE_FREE(srcPtr, &err);
        CUDA_SAFE_FREE(gpuInverse, &err);
    }
}}}
