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

namespace br { namespace cuda { namespace pca {
  void castFloatToDouble(float* a, int inca, double* b, int incb, int numElems);
  void castDoubleToFloat(double* a, int inca, float* b, int incb, int numElems);
}}}

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
    Eigen::VectorXf eVals;
    Eigen::MatrixXf eVecs;

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
      const int dimsIn = *(int*)cudaTrainingSet.first().m().ptr<void*>()[1]
                               * *(int*)cudaTrainingSet.first().m().ptr<void*>()[2];

      // copy the data over
      double* cudaDataPtr;
      CUDA_SAFE_MALLOC(&cudaDataPtr, instances*dimsIn*sizeof(cudaDataPtr[0]), &cudaError);
      for (int i=0; i < instances; i++) {
        br::cuda::pca::castFloatToDouble(
          (float*)(cudaTrainingSet[i].m().ptr<void*>()[0]),
          1,
          cudaDataPtr+i*dimsIn,
          1,
          dimsIn
        );
      }

      trainCore(cudaDataPtr, dimsIn, instances);

      CUDA_SAFE_FREE(cudaDataPtr, &cudaError);
    }

    void project(const Template &src, Template &dst) const
    {
      cudaError_t cudaError;

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
        CUBLAS_ERROR_CHECK(status);
      }

      //cout << "Saving result" << endl;
      dst = dstMat;
      CUDA_SAFE_FREE(srcGpuMatPtr, &cudaError);
    }

    void store(QDataStream &stream) const
    {
        stream << keep << drop << whiten <<  mean << eVecs;
    }

    void load(QDataStream &stream)
    {
        stream >> keep >> drop >> whiten >> mean >> eVecs;

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
    void trainCore(double* cudaDataPtr, int dimsIn, int instances) {
      cudaError_t cudaError;

      const bool dominantEigenEstimation = (dimsIn > instances);

      Eigen::MatrixXd allEVals, allEVecs;

      // allocate the eigenvectors
      if (dominantEigenEstimation) {
        allEVals = Eigen::MatrixXd(instances, 1);
        allEVecs = Eigen::MatrixXd(dimsIn, instances);
      } else {
        allEVals = Eigen::MatrixXd(dimsIn, 1);
        allEVecs = Eigen::MatrixXd(dimsIn, dimsIn);
      }

      if (keep != 0) {
        performCovarianceSVD(cudaDataPtr, dimsIn, instances, allEVals, allEVecs);
      } else {
        // null case
        mean = Eigen::VectorXf::Zero(dimsIn);
        allEVecs = Eigen::MatrixXd::Identity(dimsIn, dimsIn);
        allEVals = Eigen::VectorXd::Ones(dimsIn);
      }

      // *****************
      // We have now found the eigenvalues and eigenvectors
      // *****************

      if (keep <= 0) {
          keep = dimsIn - drop;
      } else if (keep < 1) {
          // Keep eigenvectors that retain a certain energy percentage.
          const double totalEnergy = allEVals.sum();
          if (totalEnergy == 0) {
              keep = 0;
          } else {
              double currentEnergy = 0;
              int i=0;
              while ((currentEnergy / totalEnergy < keep) && (i < allEVals.rows())) {
                  currentEnergy += allEVals(i);
                  i++;
              }
              keep = i - drop;
          }
      } else {
          if (keep + drop > allEVals.rows()) {
              qWarning("Insufficient samples, needed at least %d but only got %d.", (int)keep + drop, (int)allEVals.rows());
              keep = allEVals.rows() - drop;
          }
      }

      // Keep highest energy vectors
      eVals = Eigen::VectorXf((int)keep, 1);
      eVecs = Eigen::MatrixXf(allEVecs.rows(), (int)keep);
      for (int i=0; i<keep; i++) {
          int index = i+drop;
          eVals(i) = allEVals(index);
          eVecs.col(i) = allEVecs.col(index).cast<float>() / allEVecs.col(index).norm();
          if (whiten) eVecs.col(i) /= sqrt(eVals(i));
      }

      // Debug output
      if (Globals->verbose) qDebug() << "PCA Training:\n\tDimsIn =" << dimsIn << "\n\tKeep =" << keep;
    }

    // computes the covariance matrix and then pulls the eigenvalues+eigenvectors
    // out of it using SVD of a symmetric matrix
    void performCovarianceSVD(double* cudaDataPtr, int dimsIn, int instances, Eigen::MatrixXd& allEVals, Eigen::MatrixXd& allEVecs) {
      cudaError_t cudaError;

      const bool dominantEigenEstimation = (dimsIn > instances);

      // used for temporary storage
      Eigen::VectorXd meanDouble(dimsIn);

      // compute the mean
      for (int i=0; i < dimsIn; i++) {
        cublasDasum(
          cublasHandle,
          instances,
          cudaDataPtr+i,
          dimsIn,
          meanDouble.data()+i
        );
      }

      // put data back on GPU for further processing
      double* cudaMeanDoublePtr;
      CUDA_SAFE_MALLOC(&cudaMeanDoublePtr, dimsIn*sizeof(cudaMeanDoublePtr[0]), &cudaError);
      cublasSetVector(
        dimsIn,
        sizeof(cudaMeanDoublePtr[0]),
        meanDouble.data(),
        1,
        cudaMeanDoublePtr,
        1
      );

      // scale to calculate average
      {
        double scaleFactor = 1.0/(double)instances;
        cublasDscal(
          cublasHandle,
          dimsIn,
          &scaleFactor,
          cudaMeanDoublePtr,
          1
        );
      }

      // subtract mean from data
      for (int i=0; i < instances; i++) {
        double negativeOne = -1.0;
        cublasDaxpy(
          cublasHandle,
          dimsIn,
          &negativeOne,
          cudaMeanDoublePtr,
          1,
          cudaDataPtr+i*dimsIn,
          1
        );
      }

      // convert to float form and copy the data back
      CUDA_SAFE_MALLOC(&cudaMeanPtr, dimsIn*sizeof(cudaMeanPtr[0]), &cudaError);
      br::cuda::pca::castDoubleToFloat(cudaMeanDoublePtr, 1, cudaMeanPtr, 1, dimsIn);

      // copy the data back
      mean = Eigen::VectorXf(dimsIn);
      cublasGetVector(
        dimsIn,
        sizeof(cudaMeanPtr[0]),
        cudaMeanPtr,
        1,
        mean.data(),
        1
      );

      // free up the memory
      CUDA_SAFE_FREE(cudaMeanDoublePtr, &cudaError);
      CUDA_SAFE_FREE(cudaMeanPtr, &cudaError);

      // allocate space for the covariance matrix
      double* cudaCovariancePtr;
      int covRows = allEVals.rows();
      CUDA_SAFE_MALLOC(&cudaCovariancePtr, covRows*covRows*sizeof(cudaCovariancePtr[0]), &cudaError);

      // compute the covariance matrix
      if (dominantEigenEstimation) {
        // cov = data.transpose() * data / (instances-1.0);
        const double scaleFactor = 1.0/(instances-1.0);
        const double zero = 0.0;
        cublasDgemm(
          cublasHandle,
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          instances,
          instances,
          dimsIn,
          &scaleFactor,
          cudaDataPtr,
          dimsIn,
          cudaDataPtr,
          dimsIn,
          &zero,
          cudaCovariancePtr,
          covRows
        );
      } else {
        // cov = data * data.transpose() / (instances-1.0);
        const double scaleFactor = 1.0/(instances-1.0);
        const double zero = 0.0;
        cublasDgemm(
          cublasHandle,
          CUBLAS_OP_N,
          CUBLAS_OP_T,
          dimsIn,
          dimsIn,
          instances,
          &scaleFactor,
          cudaDataPtr,
          dimsIn,
          cudaDataPtr,
          dimsIn,
          &zero,
          cudaCovariancePtr,
          covRows
        );
      }

      cusolverDnHandle_t cusolverHandle;
      cusolverStatus_t cusolverStatus;
      cusolverDnCreate(&cusolverHandle);

      // allocate appropriate working space
      int svdLWork;
      cusolverDnDgesvd_bufferSize(
        cusolverHandle,
        covRows,
        covRows,
        &svdLWork
      );
      double* cudaSvdWork;
      CUDA_SAFE_MALLOC(&cudaSvdWork, svdLWork*sizeof(cudaSvdWork[0]), &cudaError);

      double* cudaUPtr;
      CUDA_SAFE_MALLOC(&cudaUPtr, covRows*covRows*sizeof(cudaUPtr[0]), &cudaError);
      double* cudaSPtr;
      CUDA_SAFE_MALLOC(&cudaSPtr, covRows*sizeof(cudaSPtr[0]), &cudaError);
      double* cudaVTPtr;
      CUDA_SAFE_MALLOC(&cudaVTPtr, covRows*covRows*sizeof(cudaVTPtr[0]), &cudaError);

      int* cudaSvdDevInfoPtr;
      CUDA_SAFE_MALLOC(&cudaSvdDevInfoPtr, sizeof(*cudaSvdDevInfoPtr), &cudaError);
      int svdDevInfo;

      // perform SVD on an n x m matrix, in this case the matrix is the covariance
      // matrix and is symmetric, meaning the SVD will calculate the eigenvalues
      // and eigenvectors for us.
      cusolverStatus = cusolverDnDgesvd(
        cusolverHandle,
        'A',                // all columns of unitary matrix
        'A',                // all columns of array VT
        covRows,            // m
        covRows,            // n
        cudaCovariancePtr,  // decomposing the covariance matrix
        covRows,            // lda
        cudaSPtr,           // holds S
        cudaUPtr,           // holds U
        covRows,            // ldu
        cudaVTPtr,          // holds VT
        covRows,            // ldvt
        cudaSvdWork,        // work buffer ptr
        svdLWork,           // length of the work buffer
        NULL,               // rwork, not used for real data types
        cudaSvdDevInfoPtr   // devInfo pointer
      );
      CUSOLVER_ERROR_CHECK(cusolverStatus);

      // get the eigenvalues and free memory
      cublasGetVector(
        covRows,
        sizeof(cudaSPtr[0]),
        cudaSPtr,
        1,
        allEVals.data(),
        1
      );
      CUDA_SAFE_FREE(cudaSvdWork, &cudaError);
      CUDA_SAFE_FREE(cudaSPtr, &cudaError);
      CUDA_SAFE_FREE(cudaVTPtr, &cudaError);
      CUDA_SAFE_FREE(cudaSvdDevInfoPtr, &cudaError);

      // if this is a dominant eigen estimation, then perform matrix multiplication again
      // if (dominantEigenEstimation) allEVecs = data * allEVecs;
      if (dominantEigenEstimation) {
        double* cudaMultedAllEVecs;
        CUDA_SAFE_MALLOC(&cudaMultedAllEVecs, dimsIn*instances*sizeof(cudaMultedAllEVecs[0]), &cudaError);
        const double one = 1.0;
        const double zero = 0;

        cublasDgemm(
          cublasHandle,   // handle
          CUBLAS_OP_N,    // transa
          CUBLAS_OP_N,    // transb
          dimsIn,         // m
          instances,      // n
          instances,      // k
          &one,           // alpha
          cudaDataPtr,    // A
          dimsIn,         // lda
          cudaUPtr,       // B
          instances,      // ldb
          &zero,          // beta
          cudaMultedAllEVecs, // C
          dimsIn          // ldc
        );

        // get the eigenvectors from the multiplied value
        cublasGetMatrix(
          dimsIn,
          instances,
          sizeof(cudaMultedAllEVecs[0]),
          cudaMultedAllEVecs,
          dimsIn,
          allEVecs.data(),
          dimsIn
        );

        // free the memory used for multiplication
        CUDA_SAFE_FREE(cudaMultedAllEVecs, &cudaError);
      } else {
        // get the eigenvectors straight from the SVD
        cublasGetMatrix(
          covRows,
          covRows,
          sizeof(cudaUPtr[0]),
          cudaUPtr,
          covRows,
          allEVecs.data(),
          covRows
        );
      }


      // free all the memory
      CUDA_SAFE_FREE(cudaCovariancePtr, &cudaError);
      CUDA_SAFE_FREE(cudaUPtr, &cudaError);
      cusolverDnDestroy(cusolverHandle);
    }
};

BR_REGISTER(Transform, CUBLASPCATransform)
} // namespace br

#include "cuda/cublaspca.moc"
