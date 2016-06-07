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

    Eigen::VectorXf mean, eVals;
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
      // copy the data back from the graphics card so the training can be done on the CPU
        const int instances = cudaTrainingSet.size();       // get the number of training set instances
        QList<Template> trainingQlist;
        for(int i=0; i<instances; i++) {
          Template currentTemplate = cudaTrainingSet[i];
          void* const* srcDataPtr = currentTemplate.m().ptr<void*>();
          void* cudaMemPtr = srcDataPtr[0];
          int rows = *((int*)srcDataPtr[1]);
          int cols = *((int*)srcDataPtr[2]);
          int type = *((int*)srcDataPtr[3]);

          if (type != CV_32FC1) {
            qFatal("Requires single channel 32-bit floating point matrices.");
          }

          // copy GPU mat data back to the CPU so we can do the training on the CPU
          Mat mat = Mat(rows, cols, type);
          cudaError_t err;
          CUDA_SAFE_MEMCPY(mat.ptr<float>(), cudaMemPtr, rows*cols*sizeof(float), cudaMemcpyDeviceToHost, &err);
          trainingQlist.append(Template(mat));
        }

        // assemble a TemplateList from the list of data
        TemplateList trainingSet(trainingQlist);

        originalRows = trainingSet.first().m().rows;    // get number of rows of first image
        int dimsIn = trainingSet.first().m().rows * trainingSet.first().m().cols; // get the size of the first image

        // Map into 64-bit Eigen matrix - perform the column major conversion
        Eigen::MatrixXd data(dimsIn, instances);        // create a mat
        for (int i=0; i<instances; i++) {
          data.col(i) = Eigen::Map<const Eigen::MatrixXf>(trainingSet[i].m().ptr<float>(), dimsIn, 1).cast<double>();
        }

        trainCore(data);
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
        //cout << "Ax + y" << endl;
        // subtract out the average
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
        //cout << "Matrix-Vector multiplication" << endl;

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
        stream << keep << drop << whiten << originalRows << mean << eVals << eVecs;
    }

    void load(QDataStream &stream)
    {
        stream >> keep >> drop >> whiten >> originalRows >> mean >> eVals >> eVecs;

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
    void trainCore(Eigen::MatrixXd data)
    {
        int dimsIn = data.rows();
        int instances = data.cols();
        const bool dominantEigenEstimation = (dimsIn > instances);

        Eigen::MatrixXd allEVals, allEVecs;
        if (keep != 0) {
            // Compute and remove mean
            mean = Eigen::VectorXf(dimsIn);
            for (int i=0; i<dimsIn; i++) mean(i) = data.row(i).sum() / (float)instances;
            for (int i=0; i<dimsIn; i++) data.row(i).array() -= mean(i);

            // Calculate covariance matrix
            Eigen::MatrixXd cov;
            if (dominantEigenEstimation) cov = data.transpose() * data / (instances-1.0);
            else                         cov = data * data.transpose() / (instances-1.0);

            // Compute eigendecomposition. Returns eigenvectors/eigenvalues in increasing order by eigenvalue.
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eSolver(cov);
            allEVals = eSolver.eigenvalues();
            allEVecs = eSolver.eigenvectors();
            if (dominantEigenEstimation) allEVecs = data * allEVecs;
        } else {
            // Null case
            mean = Eigen::VectorXf::Zero(dimsIn);
            allEVecs = Eigen::MatrixXd::Identity(dimsIn, dimsIn);
            allEVals = Eigen::VectorXd::Ones(dimsIn);
        }

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
                    currentEnergy += allEVals(allEVals.rows()-(i+1));
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
            int index = allEVals.rows()-(i+drop+1);
            eVals(i) = allEVals(index);
            eVecs.col(i) = allEVecs.col(index).cast<float>() / allEVecs.col(index).norm();
            if (whiten) eVecs.col(i) /= sqrt(eVals(i));
        }

        // Debug output
        if (Globals->verbose) qDebug() << "PCA Training:\n\tDimsIn =" << dimsIn << "\n\tKeep =" << keep;
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
