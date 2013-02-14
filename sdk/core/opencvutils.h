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

#ifndef __OPENCVUTILS_H
#define __OPENCVUTILS_H

#include <QDataStream>
#include <QDebug>
#include <QString>
#include <QStringList>
#include <opencv2/core/core.hpp>

namespace OpenCVUtils
{
    // Test/write/display image
    void saveImage(const cv::Mat &src, const QString &file);
    void showImage(const cv::Mat &src, const QString &window = "OpenBR", bool waitKey = true);

    // Convert image
    void cvtGray(const cv::Mat &src, cv::Mat &dst);
    void cvtUChar(const cv::Mat &src, cv::Mat &dst);

    // To image
    cv::Mat toMat(const QList<float> &src, int rows = -1);
    cv::Mat toMat(const QList<cv::Mat> &src);      // Data organized one matrix per row
    cv::Mat toMatByRow(const QList<cv::Mat> &src); // Data organized one row per row

    // From image
    QString elemToString(const cv::Mat &m, int r, int c);
    float elemToFloat(const cv::Mat &m, int r, int c);
    QString matrixToString(const cv::Mat &m);
    QStringList matrixToStringList(const cv::Mat &m);
    QList<float> matrixToVector(const cv::Mat &m);

    // Conversions
    cv::Point2f toPoint(const QPointF &qPoint);
    QPointF fromPoint(const cv::Point2f &cvPoint);
    QList<cv::Point2f> toPoints(const QList<QPointF> &qPoints);
    QList<QPointF> fromPoints(const QList<cv::Point2f> &cvPoints);
    cv::Rect toRect(const QRectF &qRect);
    QRectF fromRect(const cv::Rect &cvRect);
    QList<cv::Rect> toRects(const QList<QRectF> &qRects);
    QList<QRectF> fromRects(const QList<cv::Rect> &cvRects);

    int getFourcc();
}

QDebug operator<<(QDebug dbg, const cv::Mat &m);
QDebug operator<<(QDebug dbg, const cv::Point &p);
QDebug operator<<(QDebug dbg, const cv::Rect &r);
QDataStream &operator<<(QDataStream &stream, const cv::Mat &m);
QDataStream &operator>>(QDataStream &stream, cv::Mat &m);
QDataStream &operator<<(QDataStream &stream, const cv::Rect &r);
QDataStream &operator>>(QDataStream &stream, cv::Rect &r);
QDataStream &operator<<(QDataStream &stream, const cv::Size &s);
QDataStream &operator>>(QDataStream &stream, cv::Size &s);


// As described in "Modern C++ Design" Section 2.5.
template <typename T>
struct Type2Type
{
    typedef T OriginalType;
};

/****
OpenCVType
    Templated OpenCV Mat::type creation.
****/
template <typename Depth, int Channels>
class OpenCVType
{
    static int getDepth(Type2Type<uchar>)
    {
        return CV_8U;
    }

    static int getDepth(Type2Type<char>)
    {
        return CV_8S;
    }

    static int getDepth(Type2Type<ushort>)
    {
        return CV_16U;
    }

    static int getDepth(Type2Type<short>)
    {
        return CV_16S;
    }

    static int getDepth(Type2Type<long>)
    {
        return CV_32S;
    }

    static int getDepth(Type2Type<float>)
    {
        return CV_32F;
    }

    static int getDepth(Type2Type<double>)
    {
        return CV_64F;
    }

public:
    static int make()
    {
        return CV_MAKETYPE((getDepth(Type2Type<Depth>())),(Channels));
    }
};

#endif // __OPENCVUTILS_H
