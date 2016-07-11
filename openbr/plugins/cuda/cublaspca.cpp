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
#include <unistd.h>

#include <QList>

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
using namespace cv;

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/common.h>
#include <openbr/core/eigenutils.h>
#include <openbr/core/opencvutils.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "cudadefines.hpp"

namespace br
{
/*!
 * \ingroup transforms
 * \brief Projects input into learned Principal Component Analysis subspace using CUDA. Modified from original PCA plugin.
 * \author Colin Heinzmann \cite DepthDeluxe
 *
 * \br_property float keep Options are: [keep < 0 - All eigenvalues are retained, keep == 0 - No PCA is performed and the eigenvectors form an identity matrix, 0 < keep < 1 - Keep is the fraction of the variance to retain, keep >= 1 - keep is the number of leading eigenvectors to retain] Default is 0.95.
 * \br_property int drop The number of leading eigen-dimensions to drop.
 * \br_property bool whiten Whether or not to perform PCA whitening (i.e., normalize variance of each dimension to unit norm)
 */
class CUBLASPCATransform : public Transform
{
    Q_OBJECT

protected:
    Q_PROPERTY(float keep READ get_keep WRITE set_keep RESET reset_keep STORED false)
    Q_PROPERTY(int drop READ get_drop WRITE set_drop RESET reset_drop STORED false)
    Q_PROPERTY(bool whiten READ get_whiten WRITE set_whiten RESET reset_whiten STORED false)

    BR_PROPERTY(float, keep, 0.95)
    BR_PROPERTY(int, drop, 0)
    BR_PROPERTY(bool, whiten, false)

    Eigen::VectorXf mean;
    Eigen::MatrixXf eVecs;

    int originalRows;

    cublasHandle_t cublasHandle;
    float* cudaMeanPtr;     // holds the "keep" long vector
    float* cudaEvPtr;       // holds all the eigenvectors

public:
    CUBLASPCATransform() : keep(0.95), drop(0), whiten(false) {
      // try to initialize CUBLAS
      cublasStatus_t status;
      status = cublasCreate(&cublasHandle);
      CUBLAS_ERROR_CHECK(status);
    }

    ~CUBLASPCATransform() {
      // tear down CUBLAS
      cublasDestroy(cublasHandle);
    }

private:
    double residualReconstructionError(const Template &src) const
    {
        Template proj;
        project(src, proj);

        Eigen::Map<const Eigen::VectorXf> srcMap(src.m().ptr<float>(), src.m().rows*src.m().cols);
        Eigen::Map<Eigen::VectorXf> projMap(proj.m().ptr<float>(), keep);

        return (srcMap - mean).squaredNorm() - projMap.squaredNorm();
    }

    void train(const TemplateList &cudaTrainingSet)
    {
      cublasStatus_t cublasStatus;
      cudaError_t cudaError;

      // put all the data into a single matrix to perform PCA
      const int instances = cudaTrainingSet.size();
      const int instanceSize = *(int*)cudaTrainingSet.first().m().ptr<void*>()[1]
                               * *(int*)cudaTrainingSet.first().m().ptr<void*>()[2];

      // get all the vectors from memory
      Eigen::MatrixXf data(instanceSize, instances);
      for (int i=0; i < instances; i++) {
        float* currentCudaMatPtr = (float*)cudaTrainingSet[i].m().ptr<void*>()[0];

        cublasGetVector(
          instanceSize,
          sizeof(float),
          currentCudaMatPtr,
          1,
          data.data()+i*instanceSize,
          1
        );
      }

      Eigen::MatrixXd dataDouble(instanceSize, instances);
      for (int i=0; i < instanceSize*instances; i++) {
        dataDouble.data()[i] = (double)data.data()[i];
      }

      // XXX: remove me
      Eigen::MatrixXd test(3,3);
      test << 1,2,3,4,5,6,7,8,9;
      trainCore(test);

      // trainCore(dataDouble);
    }

    void project(const Template &src, Template &dst) const
    {
      //cout << "Starting projection" << endl;

      void* const* srcDataPtr = src.m().ptr<void*>();
      float* srcGpuMatPtr = (float*)srcDataPtr[0];
      int rows = *((int*)srcDataPtr[1]);
      int cols = *((int*)srcDataPtr[2]);
      int type = *((int*)srcDataPtr[3]);

      if (type != CV_32FC1) {
        cout << "ERR: Invalid image type" << endl;
        throw 0;
      }

      // save the destination rows
      int dstRows = (int)keep;

      Mat dstMat = Mat(src.m().rows, src.m().cols, src.m().type());
      void** dstDataPtr = dstMat.ptr<void*>();
      float** dstGpuMatPtrPtr = (float**)dstDataPtr;
      dstDataPtr[1] = srcDataPtr[1];  *((int*)dstDataPtr[1]) = 1;
      dstDataPtr[2] = srcDataPtr[2];  *((int*)dstDataPtr[2]) = dstRows;
      dstDataPtr[3] = srcDataPtr[3];


      // allocate the memory and set to zero
      //cout << "Allocating destination memory" << endl;
      cublasStatus_t status;
      cudaMalloc(dstGpuMatPtrPtr, dstRows*sizeof(float));
      cudaMemset(*dstGpuMatPtrPtr, 0, dstRows*sizeof(float));

      {
        float negativeOne = -1.0f;
        status = cublasSaxpy(
          cublasHandle,       // handle
          dstRows,            // vector length
          &negativeOne,       // alpha (1)
          cudaMeanPtr,        // mean
          1,                  // stride
          srcGpuMatPtr,       // y, the source
          1                   // stride
        );
        CUBLAS_ERROR_CHECK(status);
      }

      {
        float one = 1.0f;
        float zero = 0.0f;
        status = cublasSgemv(
          cublasHandle,       // handle
          CUBLAS_OP_T,        // normal vector multiplication
          eVecs.rows(),       // # rows
          eVecs.cols(),       // # cols
          &one,               // alpha (1)
          cudaEvPtr,          // pointer to the matrix
          eVecs.rows(),       // leading dimension of matrix
          srcGpuMatPtr,       // vector for multiplication
          1,                  // stride (1)
          &zero,              // beta (0)
          *dstGpuMatPtrPtr,   // vector to store the result
          1                   // stride (1)
        );
      }

      //cout << "Saving result" << endl;
      dst = dstMat;
      cudaFree(srcGpuMatPtr);
    }

    void store(QDataStream &stream) const
    {
        stream << keep << drop << whiten << originalRows << mean << eVecs;
    }

    void load(QDataStream &stream)
    {
        stream >> keep >> drop >> whiten >> originalRows >> mean >>  eVecs;

        //cout << "Starting load process" << endl;

        cudaError_t cudaError;
        cublasStatus_t cublasStatus;
        CUDA_SAFE_MALLOC(&cudaMeanPtr, mean.rows()*mean.cols()*sizeof(float), &cudaError);
        CUDA_SAFE_MALLOC(&cudaEvPtr, eVecs.rows()*eVecs.cols()*sizeof(float), &cudaError);

        //cout << "Setting vector" << endl;
        // load the mean vector into GPU memory
        cublasStatus = cublasSetVector(
          mean.rows()*mean.cols(),
          sizeof(float),
          mean.data(),
          1,
          cudaMeanPtr,
          1
        );
        CUBLAS_ERROR_CHECK(cublasStatus);

        //cout << "Setting the matrix" << endl;
        // load the eigenvector matrix into GPU memory
        cublasStatus = cublasSetMatrix(
          eVecs.rows(),
          eVecs.cols(),
          sizeof(float),
          eVecs.data(),
          eVecs.rows(),
          cudaEvPtr,
          eVecs.rows()
        );
        CUBLAS_ERROR_CHECK(cublasStatus);
    }

protected:
    void trainCore(Eigen::MatrixXd data) {
      cudaError_t cudaError;

      // utility variables
      const double one = 1.0;
      const double negativeOne = -1.0;
      const double zero = 0.0;

      static int numTimesThrough = 0;
      numTimesThrough++;

      int dimsIn = data.rows();       // the number of rows of the covariance matrix
      int instances = data.cols();    // the number of columns of the covariance matrix
      const bool dominantEigenEstimation = (dimsIn > instances);

      // Compute and remove mean
      //mean = Eigen::VectorXf(dimsIn);
      //for (int i=0; i<dimsIn; i++) mean(i) = data.row(i).sum() / (float)instances;
      //for (int i=0; i<dimsIn; i++) data.row(i).array() -= mean(i);

      // allocate and place data in GPU memory
      double* cudaDataPtr;
      CUDA_SAFE_MALLOC(&cudaDataPtr, data.rows()*data.cols()*sizeof(cudaDataPtr[0]), &cudaError);
      cublasSetMatrix(
        data.rows(),
        data.cols(),
        sizeof(cudaDataPtr[0]),
        data.data(),
        data.rows(),
        cudaDataPtr,
        data.rows()
      );

      // allocate space for the covariance matrix
      double* cudaCovariancePtr;
      int covRows = data.cols();
      CUDA_SAFE_MALLOC(&cudaCovariancePtr, covRows*covRows*sizeof(cudaCovariancePtr[0]), &cudaError);

      // compute the covariance matrix
      // cov = data.transpose() * data / (instances-1.0);
      {
        double scaleFactor = 1.0/(instances-1.0);
        cublasDgemm(
          cublasHandle,
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          data.cols(),
          data.cols(),
          data.rows(),
          &scaleFactor,
          cudaDataPtr,
          data.rows(),
          cudaDataPtr,
          data.rows(),
          &zero,
          cudaCovariancePtr,
          covRows
        );
      }

      // XXX: download the covariace matrix for debugging
      Eigen::MatrixXd cov(covRows, covRows);
      cublasGetMatrix(
        covRows,
        covRows,
        sizeof(cov.data()[0]),
        cudaCovariancePtr,
        covRows,
        cov.data(),
        covRows
      );

      // initialize cuSolver for the next part
      cusolverDnHandle_t cusolverHandle;
      cusolverDnCreate(&cusolverHandle);

      double* cudaEigenvaluesPtr;
      {
        double* cudaDiagonalPtr;
        CUDA_SAFE_MALLOC(&cudaDiagonalPtr, covRows*sizeof(double), &cudaError);
        double* cudaOffdiagonalPtr;
        CUDA_SAFE_MALLOC(&cudaOffdiagonalPtr, covRows*sizeof(double), &cudaError);
        double* cudaTauqPtr;
        CUDA_SAFE_MALLOC(&cudaTauqPtr, covRows*sizeof(double), &cudaError);
        double* cudaTaupPtr;
        CUDA_SAFE_MALLOC(&cudaTaupPtr, covRows*sizeof(double), &cudaError);

        // calculate lwork
        int lwork;
        cusolverDnSgebrd_bufferSize(
          cusolverHandle,
          covRows,
          covRows,
          &lwork
        );
        double* cudaWorkBufferPtr;
        CUDA_SAFE_MALLOC(&cudaWorkBufferPtr, lwork, &cudaError);

        int* cudaDevInfoPtr;
        CUDA_SAFE_MALLOC(&cudaDevInfoPtr, sizeof(int), &cudaError);

        // call the eigenvalue decomposer
        cusolverDnDgebrd(
          cusolverHandle,
          covRows,
          covRows,
          cudaCovariancePtr,
          covRows,
          cudaDiagonalPtr,
          cudaOffdiagonalPtr,
          cudaTauqPtr,
          cudaTaupPtr,
          cudaWorkBufferPtr,
          lwork,
          cudaDevInfoPtr
        );

        // the eigenvalues are on the diagonal
        cudaEigenvaluesPtr = cudaOffdiagonalPtr;

        /*
        // initialize the result buffers
        double* cudaOffdiagonalPtr;
        CUDA_SAFE_MALLOC(&cudaEigenvaluesPtr, covRows*sizeof(cudaEigenvaluesPtr[0]), &cudaError);
        CUDA_SAFE_MALLOC(&cudaOffdiagonalPtr, covRows*sizeof(cudaOffdiagonalPtr[0]), &cudaError);

        // initialize the tauq and taup buffers
        double* cudaTauqPtr;
        double* cudaTaupPtr;
        CUDA_SAFE_MALLOC(&cudaTauqPtr, covRows*sizeof(cudaTauqPtr[0]), &cudaError);
        CUDA_SAFE_MALLOC(&cudaTaupPtr, covRows*sizeof(cudaTaupPtr[0]), &cudaError);

        // build the work buffer
        double* cudaWorkPtr;
        int workBufferSize;
        cusolverDnSgesvd_bufferSize(cusolverHandle, covRows, covRows, &workBufferSize);
        CUDA_SAFE_MALLOC(&cudaWorkPtr, workBufferSize, &cudaError);

        int* cudaDevInfoPtr;
        CUDA_SAFE_MALLOC(&cudaDevInfoPtr, sizeof(*cudaDevInfoPtr), &cudaError);

        // now pull the eigenvalues out
        cusolverStatus_t cusolverStatus;
        cusolverStatus = cusolverDnDgebrd(
          cusolverHandle,           // handle
          covRows,                  // rows of Matrix A
          covRows,                  // cols of Matrix A
          cudaCovariancePtr,        // CUDA pointer to matrix A
          covRows,                  // leading dimension of A
          cudaEigenvaluesPtr,       // diagonal elements of bidiagonal matrix
          cudaOffdiagonalPtr,       // off-diagonal elements of matrix
          cudaTauqPtr,
          cudaTaupPtr,
          cudaWorkPtr,
          workBufferSize,
          cudaDevInfoPtr
        );

        // print out the devInfo
        int devInfo;
        cudaMemcpy(&devInfo, cudaDevInfoPtr, sizeof(devInfo), cudaMemcpyDeviceToHost);

        // now we have the eigenvalues

        // XXX: the off diagonal values
        Eigen::VectorXd offDiagonal(covRows);
        cublasGetVector(
          covRows,
          sizeof(offDiagonal.data()[0]),
          cudaOffdiagonalPtr,
          1,
          offDiagonal.data(),
          1
        );

        // clean up
        CUDA_SAFE_FREE(cudaOffdiagonalPtr, &cudaError);
        CUDA_SAFE_FREE(cudaTauqPtr, &cudaError);
        CUDA_SAFE_FREE(cudaTaupPtr, &cudaError);
        CUDA_SAFE_FREE(cudaWorkPtr, &cudaError);
        CUDA_SAFE_FREE(cudaDevInfoPtr, &cudaError);
        */
      }

      // copy the eigenvalues back to the CPU
      Eigen::VectorXd allEVals(covRows);
      cublasGetVector(
        covRows,
        sizeof(allEVals.data()[0]),
        cudaEigenvaluesPtr,
        1,
        allEVals.data(),
        1
      );
      CUDA_SAFE_FREE(cudaEigenvaluesPtr, &cudaError);

      // now find the eigenvectors
      Eigen::MatrixXd allEVecs(covRows, covRows);
      double* cudaCoefficientMatrix;
      CUDA_SAFE_MALLOC(&cudaCoefficientMatrix, covRows*covRows*sizeof(cudaCoefficientMatrix[0]), &cudaError);
      for (int i=0; i < covRows; i++) {
        // load cov into matrix
        cublasSetMatrix(
          covRows,
          covRows,
          sizeof(cov.data()[0]),
          cov.data(),
          covRows,
          cudaCoefficientMatrix,
          covRows
        );

        // subtract out the Eigenvalue from the center of the matrix
        // first copy the eigenvalue into a single buffer
        double* cudaEigenvalueSubtractBuffer;
        CUDA_SAFE_MALLOC(&cudaEigenvalueSubtractBuffer, covRows*sizeof(cudaEigenvalueSubtractBuffer[0]), &cudaError);
        for(int j = 0; j < covRows; j++) {
          cublasSetVector(
            1,
            sizeof(cudaEigenvalueSubtractBuffer[0]),
            &allEVals.data()[i],
            1,
            &cudaEigenvalueSubtractBuffer[j],
            1
          );
        }

        // perform the subtraction
        cublasDaxpy(
          cublasHandle,
          covRows,
          &negativeOne,
          cudaEigenvalueSubtractBuffer,
          1,
          cudaCoefficientMatrix,
          covRows+1                     // move across the diagonal
        );

        // perform the Cholesky factorization of the coefficient matrix
        double* cudaWorkBufferPtr;
        int lwork;
        cusolverDnDpotrf_bufferSize(
          cusolverHandle,
          CUBLAS_FILL_MODE_UPPER,
          covRows,
          cudaCoefficientMatrix,
          covRows,
          &lwork
        );
        CUDA_SAFE_MALLOC(&cudaWorkBufferPtr, lwork, &cudaError);

        int* cudaDevInfoPtr;
        CUDA_SAFE_MALLOC(&cudaDevInfoPtr, sizeof(*cudaDevInfoPtr), &cudaError);

        cusolverDnDpotrf(
          cusolverHandle,
          CUBLAS_FILL_MODE_UPPER,
          covRows,
          cudaCoefficientMatrix,
          covRows,
          cudaWorkBufferPtr,
          lwork,
          cudaDevInfoPtr
        );
        int devInfo;
        CUDA_SAFE_MEMCPY(&devInfo, cudaDevInfoPtr, sizeof(devInfo), cudaMemcpyDeviceToHost, &cudaError);
        cout << "DevInfo: " << devInfo << endl;


        // XXX: remove after dbugging
        Eigen::MatrixXd anotherMatrix(covRows, covRows);
        cublasGetMatrix(
          covRows,
          covRows,
          sizeof(anotherMatrix.data()[0]),
          cudaCoefficientMatrix,
          1,
          anotherMatrix.data(),
          1
        );

        // the first element of B is equal to the covariance matrix, the rest are zeroes
        double* cudaBVector;
        CUDA_SAFE_MALLOC(&cudaBVector, covRows*sizeof(cudaBVector[0]), &cudaError);
        // load the top element to be the same as first of coefficient
        // this results in the first variable being zero and assigning
        // values for the rest of the matrix
        cublasDcopy(
          cublasHandle,
          1,
          cudaCoefficientMatrix,
          1,
          cudaBVector,
          1
        );
        // load the rest 0's
        for (int j = 1; j < covRows; j++) {
          cublasSetVector(
            1,
            sizeof(cudaBVector[0]),
            &zero,
            1,
            &cudaBVector[j],
            1
          );
        }

        // solve the system of linear equations
        cusolverDnDpotrs(
          cusolverHandle,
          CUBLAS_FILL_MODE_LOWER,
          covRows,
          1,                        // we are solving a single system of equations
          cudaCoefficientMatrix,
          covRows,
          cudaBVector,
          covRows,
          cudaDevInfoPtr
        );
        CUDA_SAFE_MEMCPY(&devInfo, cudaDevInfoPtr, sizeof(devInfo), cudaMemcpyDeviceToHost, &cudaError);
        cout << "DevInfo: " << devInfo << endl;

        // should have the solution
        Eigen::VectorXd solutionVector(covRows);
        cublasGetVector(
          covRows,
          sizeof(solutionVector.data()[0]),
          solutionVector.data(),
          1,
          cudaBVector,
          1
        );

        cout << "solution: [";
        for (int i=0; i < covRows; i++) {
          cout << solutionVector.data()[i] << ", ";
        }
        cout << "];" << endl;

      }

      // Keep eigenvectors that retain a certain energy percentage.
      const float totalEnergy = allEVals.sum();
      if (totalEnergy == 0) {
          keep = 0;
      } else {
          float currentEnergy = 0;
          int i=0;
          while ((currentEnergy / totalEnergy < keep) && (i < allEVals.rows())) {
              currentEnergy += allEVals(allEVals.rows()-(i+1));
              i++;
          }
          keep = i - drop;
      }

      // Keep highest energy vectors
      Eigen::VectorXf eVals = Eigen::VectorXf((int)keep, 1);
      for (int i=0; i<keep; i++) {
          int index = allEVals.rows()-(i+drop+1);
          eVals(i) = allEVals(index);
      }

      cusolverDnDestroy(cusolverHandle);
      cout << "DONE" << endl;
    }


    void writeEigenVectors(const Eigen::MatrixXd &allEVals, const Eigen::MatrixXd &allEVecs) const
    {
        const int originalCols = mean.rows() / originalRows;

        { // Write out mean image
            cv::Mat out(originalRows, originalCols, CV_32FC1);
            Eigen::Map<Eigen::MatrixXf> outMap(out.ptr<float>(), mean.rows(), 1);
            outMap = mean.col(0);
            // OpenCVUtils::saveImage(out, Globals->Debug+"/PCA/eigenVectors/mean.png");
        }

        // Write out sample eigen vectors (16 highest, 8 lowest), filename = eigenvalue.
        for (int k=0; k<(int)allEVals.size(); k++) {
            if ((k < 8) || (k >= (int)allEVals.size()-16)) {
                cv::Mat out(originalRows, originalCols, CV_64FC1);
                Eigen::Map<Eigen::MatrixXd> outMap(out.ptr<double>(), mean.rows(), 1);
                outMap = allEVecs.col(k);
                // OpenCVUtils::saveImage(out, Globals->Debug+"/PCA/eigenVectors/"+QString::number(allEVals(k),'f',0)+".png");
            }
        }
    }
};

BR_REGISTER(Transform, CUBLASPCATransform)
} // namespace br

#include "cuda/cublaspca.moc"
