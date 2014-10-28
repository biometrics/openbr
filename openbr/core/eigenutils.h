/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2012 The MITRE Corporation                                      *
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

#ifndef EIGENUTILS_H
#define EIGENUTILS_H

#include <QDataStream>
#include <Eigen/Core>
#include <assert.h>

#include <opencv2/core/core.hpp>

void writeEigen(Eigen::MatrixXf X, QString filename);
void writeEigen(Eigen::MatrixXd X, QString filename);
void writeEigen(Eigen::VectorXd X, QString filename);
void writeEigen(Eigen::VectorXf X, QString filename);
void printEigen(Eigen::MatrixXd X);
void printEigen(Eigen::MatrixXf X);
void printSize(Eigen::MatrixXf X);

//Converts x y points in a single vector to two column matrix
Eigen::MatrixXf vectorToMatrix(const Eigen::MatrixXf vector);
Eigen::MatrixXf matrixToVector(const Eigen::MatrixXf matrix);

//Remove row and column from the matrix:
Eigen::MatrixXf removeRowCol(const Eigen::MatrixXf X, int row, int col);

//Convert a point list into a matrix:
Eigen::MatrixXf pointsToMatrix(const QList<QPointF> points, bool isAffine=false);
QList<QPointF> matrixToPoints(const Eigen::MatrixXf P);

//Convert cv::Mat to Eigen
Eigen::MatrixXf toEigen(const cv::Mat m);

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
inline QDataStream &operator<<(QDataStream &stream, const Eigen::Matrix< _Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols > &mat)
{
    int r = mat.rows();
    int c = mat.cols();
    stream << r << c;

    _Scalar *data = new _Scalar[r*c];
    for (int i=0; i<r; i++)
        for (int j=0; j<c; j++)
            data[i*c+j] = mat(i, j);
    int bytes = r*c*sizeof(_Scalar);
    int bytes_written = stream.writeRawData((const char*)data, bytes);
    if (bytes != bytes_written) qFatal("EigenUtils.h operator<< failure.");

    delete[] data;
    return stream;
}

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
inline QDataStream &operator>>(QDataStream &stream, Eigen::Matrix< _Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols > &mat)
{
    int r, c;
    stream >> r >> c;
    mat.resize(r, c);

    _Scalar *data = new _Scalar[r*c];
    int bytes = r*c*sizeof(_Scalar);
    int bytes_read = stream.readRawData((char*)data, bytes);
    if (bytes != bytes_read) qFatal("EigenUtils.h operator>> failure.");
    for (int i=0; i<r; i++)
        for (int j=0; j<c; j++)
            mat(i, j) = data[i*c+j];

    delete[] data;
    return stream;
}

/*Compute the mean of the each column (dim == 1) or row (dim == 2)
  of the matrix*/
template<typename T>
Eigen::MatrixBase<T> eigMean(const Eigen::MatrixBase<T>& x,int dim)
{
    if (dim == 1) {
        Eigen::MatrixBase<T> y(1,x.cols());
        for (int i = 0; i < x.cols(); i++)
            y(i) = x.col(i).sum() / x.rows();
        return y;
    } else if (dim == 2) {
        Eigen::MatrixBase<T> y(x.rows(),1);
        for (int i = 0; i < x.rows(); i++)
            y(i) = x.row(i).sum() / x.cols();
        return y;
    }
    qFatal("A matrix can only have two dimensions");
}

/*Compute the element-wise mean*/
float eigMean(const Eigen::MatrixXf& x);
/*Compute the element-wise mean*/
float eigStd(const Eigen::MatrixXf& x);

/*Compute the std dev of the each column (dim == 1) or row (dim == 2)
  of the matrix*/
template<typename T>
Eigen::MatrixBase<T> eigStd(const Eigen::MatrixBase<T>& x,int dim)
{
    Eigen::MatrixBase<T> mean = eigMean(x, dim);
    if (dim == 1) {
        Eigen::MatrixBase<T> y(1,x.cols());
        for (int i = 0; i < x.cols(); i++) {
            T value = 0;
            for (int j = 0; j < x.rows(); j++)
                value += pow(y(j, i) - mean(i), 2);
            y(i) = sqrt(value / (x.rows() - 1));
        }
        return y;
    } else if (dim == 2) {
        Eigen::MatrixBase<T> y(x.rows(),1);
        for (int i = 0; i < x.rows(); i++) {
            T value = 0;
            for (int j = 0; j < x.cols(); j++)
                value += pow(y(i, j) - mean(j), 2);
            y(i) = sqrt(value / (x.cols() - 1));
        }
        return y;
    }
    qFatal("A matrix can only have two dimensions");
}

#endif // EIGENUTILS_H
