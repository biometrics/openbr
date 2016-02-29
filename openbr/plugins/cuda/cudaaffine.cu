#include <iostream>
using namespace std;

#include <opencv2/gpu/gpu.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <math.h>

#include "cudadefines.hpp"

using namespace cv;
using namespace cv::gpu;

namespace br { namespace cuda {

    __device__ __forceinline__ uint8_t cudaaffine_kernel_get_pixel_value(int row, int col, uint8_t* srcPtr, int rows, int cols) {
        // don't do anything if the index is out of bounds
        if (row < 1 || row >= rows-1 || col < 1 || col >= cols-1) {
            if (row >= rows || col >= cols) {
                return 0;
            } else{
                return 0; }
        }
        return (srcPtr + row*cols)[col];
    }

    /*
     * trans_inv          - A pointer to a one-dimensional representation of the inverse of the transform matrix 3x3
     * dst_row            - The destination row (mapping to this row)
     * dst_col            - The destination column (mapping to this column)
     * src_row            - The computed source pixel row (mapping from this row)
     * src_col            - The computed source pixel column (mapping from this col)
     */
    __device__ __forceinline__ void cudaaffine_kernel_get_src_coord(double *trans_inv, int dst_row, int dst_col, int* src_row, int* src_col){
        *src_col = round(dst_col * trans_inv[0] + dst_row * trans_inv[3] + trans_inv[6]);
        *src_row = round(dst_col * trans_inv[1] + dst_row * trans_inv[4] + trans_inv[7]);

        //printf("Dst: [%d, %d, 1] = [%d, %d, 1] \n[ %0.4f, %0.4f, %0.4f] \n[ %0.4f, %0.4f, %0.4f ]\n[ %0.4f, %0.4f, %0.4f ]\n\n", *src_col, *src_row, dst_col, dst_row, trans_inv[0], trans_inv[1], trans_inv[2], trans_inv[3], trans_inv[4], trans_inv[5], trans_inv[6], trans_inv[7], trans_inv[8]);

		}
				

    __global__ void cudaaffine_kernel(uint8_t* srcPtr, uint8_t* dstPtr, double* trans_inv, int src_rows, int src_cols, int dst_rows, int dst_cols){
        int dstRowInd = blockIdx.y*blockDim.y+threadIdx.y;
        int dstColInd = blockIdx.x*blockDim.x+threadIdx.x;
        int dstIndex = dstRowInd*dst_cols + dstColInd;

        //printf("Kernel Inv:\n[%0.4f %0.4f %0.4f]\n[%0.4f %0.4f %0.4f]\n[%0.4f %0.4f %0.4f]\n\n", trans_inv[0], trans_inv[1], trans_inv[2], trans_inv[3], trans_inv[4], trans_inv[5], trans_inv[6], trans_inv[7], trans_inv[8]);

        int srcRowInd;
        int srcColInd;

        // don't do anything if the index is out of bounds
        if (dstRowInd < 1 || dstRowInd >= dst_rows-1 || dstColInd < 1 || dstColInd >= dst_cols-1) {
            if (dstRowInd >= dst_rows || dstColInd >= dst_cols) {
                return;
            } else{
                dstPtr[dstIndex] = 0;
                return;
            }
        }

        cudaaffine_kernel_get_src_coord(trans_inv, dstRowInd, dstColInd, &srcRowInd, &srcColInd);
        const uint8_t cval = cudaaffine_kernel_get_pixel_value(srcRowInd, srcColInd, srcPtr, src_rows, src_cols); // Get initial pixel value

        dstPtr[dstIndex] = cval;
    }

    void cudaaffine_wrapper(void* srcPtr, void** dstPtr, Mat affineTransform, int src_rows, int src_cols, int dst_rows, int dst_cols) {
        cudaError_t err;
        double* gpuInverse;

        dim3 threadsPerBlock(8, 8);
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
        // double a23 = 0;
        // double a13 = 0;
        // double a33 = 1;

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

        // Move from affineTransform to gpuAffine (currently fake)
        // double fakeAffine[6];
        // fakeAffine[0] = affineTransform.at<double>(0, 0);
        // fakeAffine[1] = affineTransform.at<double>(0, 1);
        // fakeAffine[2] = affineTransform.at<double>(0, 2);
        // fakeAffine[3] = affineTransform.at<double>(1, 0);
        // fakeAffine[4] = affineTransform.at<double>(1, 1);
        // fakeAffine[5] = affineTransform.at<double>(1, 2);

        // printf("\n");
        // printf("%f\t%f\t%f\n", a11, a12, 0.0);
        // printf("%f\t%f\t%f\n", a21, a22, 0.0);
        // printf("%f\t%f\t%f\n", a31, a32, 1.0);
        // printf("\n");

        // printf("Affine Inverse:\n");
        // for(int i = 0; i < 3; i++){
            // for(int j = 0; j < 3; j++){
                // printf("%f\t", affineInverse[3*i + j]);
            // }
            // printf("\n");
        // }


        CUDA_SAFE_MALLOC(dstPtr, dst_rows*dst_cols*sizeof(uint8_t), &err);
        CUDA_SAFE_MALLOC(&gpuInverse, 3*3*sizeof(double), &err);

        CUDA_SAFE_MEMCPY(gpuInverse, affineInverse, 9*sizeof(double), cudaMemcpyHostToDevice, &err);

        cudaaffine_kernel<<<numBlocks, threadsPerBlock>>>((uint8_t*)srcPtr, (uint8_t*)(*dstPtr), gpuInverse, src_rows, src_cols, dst_rows, dst_cols);
        CUDA_KERNEL_ERR_CHK(&err);

        CUDA_SAFE_FREE(srcPtr, &err);
        CUDA_SAFE_FREE(gpuInverse, &err);

        // printf("\n\n");
        // for(int i = 0; i < cols; i++){
            // for(int j = 0; j < src_rows; j++){
                // printf("%4d\t", ((uint8_t*) dstPtr)[j*cols + i]);
            // }
            // printf("\n");
        // }
        // printf("\n");
    }
} // end cuda
} // end br
