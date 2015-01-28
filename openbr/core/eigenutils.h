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
#include <QDebug>
#include <Eigen/Core>
#include <assert.h>

#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>

#include "openbr/core/qtutils.h"

namespace EigenUtils
{
    template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    QString matrixToString(const Eigen::Matrix< _Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols > &mat)
    {
        QString result;
        if (mat.rows() > 1) result += "{ ";
        for (int r=0; r<mat.rows(); r++) {
            if ((mat.rows() > 1) && (r > 0)) result += "  ";
            if (mat.cols() > 1) result += "[";
            for (int c=0; c<mat.cols(); c++) {
                result += QString::number(mat(r, c));
                if (c < mat.cols() - 1) result += ", ";
            }
            if (mat.cols() > 1) result += "]";
            if (r < mat.rows()-1) result += "\n";
        }
        if (mat.rows() > 1) result += " }";
        return result;
    }

    template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    void writeMatrix(const Eigen::Matrix< _Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols > &mat, const QString &filename)
    {
        int r = mat.rows();
        int c = mat.cols();

        _Scalar *data = new _Scalar[r*c];
        for (int i=0; i<r; i++)
            for (int j=0; j<c; j++)
                data[i*c+j] = mat(i, j);
        int bytes = r*c*sizeof(_Scalar);
        QByteArray byteArray((const char*)data,bytes);
        QtUtils::writeFile(filename,byteArray);
    }

    void printSize(Eigen::MatrixXf X);

    // Converts x y points in a single vector to two column matrix
    Eigen::MatrixXf vectorToMatrix(const Eigen::MatrixXf vector);
    Eigen::MatrixXf matrixToVector(const Eigen::MatrixXf matrix);

    // Remove row and column from the matrix:
    Eigen::MatrixXf removeRowCol(const Eigen::MatrixXf X, int row, int col);

    // Convert a point list into a matrix:
    Eigen::MatrixXf pointsToMatrix(const QList<QPointF> points, bool isAffine=false);
    QList<QPointF> matrixToPoints(const Eigen::MatrixXf P);

    // Compute the element-wise standard deviation
    float stddev(const Eigen::MatrixXf& x);
}

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
inline QDebug operator<<(QDebug dbg, const Eigen::Matrix< _Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols > &mat)
{
    dbg.nospace() << EigenUtils::matrixToString(mat);
    return dbg.space();
}

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

#endif // EIGENUTILS_H
