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

#ifndef OPENCVUTILS_OPENCVUTILS_H
#define OPENCVUTILS_OPENCVUTILS_H

#include <QDataStream>
#include <QDebug>
#include <QString>
#include <QStringList>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <assert.h>
#include <openbr/openbr_plugin.h>

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
    cv::Mat toMat(const QList< QList<float> > &srcs, int rows = -1);
    BR_EXPORT cv::Mat toMat(const QList<int> &src, int rows = -1);

    BR_EXPORT cv::Mat toMat(const QList<cv::Mat> &src);      // Data organized one matrix per row
    cv::Mat toMatByRow(const QList<cv::Mat> &src); // Data organized one row per row

    // From image
    QString depthToString(const cv::Mat &m);
    QString typeToString(const cv::Mat &m);
    QString elemToString(const cv::Mat &m, int r, int c);
    QString matrixToString(const cv::Mat &m);
    QStringList matrixToStringList(const cv::Mat &m);

    // Model storage
    void storeModel(const CvStatModel &model, QDataStream &stream);
    void storeModel(const cv::Algorithm &model, QDataStream &stream);
    void loadModel(CvStatModel &model, QDataStream &stream);
    void loadModel(cv::Algorithm &model, QDataStream &stream);

    template <typename T>
    T getElement(const cv::Mat &m, int r, int c)
    {
        assert(m.channels() == 1);
        switch (m.depth()) {
          case CV_8U:  return T(m.at<quint8>(r,c));
          case CV_8S:  return T(m.at<qint8>(r,c));
          case CV_16U: return T(m.at<quint16>(r,c));
          case CV_16S: return T(m.at<qint16>(r,c));
          case CV_32S: return T(m.at<qint32>(r,c));
          case CV_32F: return T(m.at<float>(r,c));
          case CV_64F: return T(m.at<double>(r,c));
          default:     qFatal("Unknown matrix depth!");
        }
        return 0;
    }

    template <typename T>
    QList<T> matrixToVector(const cv::Mat &m)
    {
        QList<T> results;
        std::vector<cv::Mat> mv;
        cv::split(m, mv);
        foreach (const cv::Mat &mc, mv)
            for (int i=0; i<mc.rows; i++)
                for (int j=0; j<mc.cols; j++)
                    results.append(getElement<float>(mc, i, j));
        return results;
    }

    // Conversions
    cv::Point2f toPoint(const QPointF &qPoint);
    QPointF fromPoint(const cv::Point2f &cvPoint);
    QList<cv::Point2f> toPoints(const QList<QPointF> &qPoints);
    QList<QPointF> fromPoints(const QList<cv::Point2f> &cvPoints);
    cv::Mat pointsToMatrix(const QList<QPointF> &qPoints);
    cv::Rect toRect(const QRectF &qRect);
    cv::RotatedRect toRotatedRect(const QRectF &qRect, float angle);
    QRectF fromRect(const cv::Rect &cvRect);
    QList<cv::Rect> toRects(const QList<QRectF> &qRects);
    QList<QRectF> fromRects(const QList<cv::Rect> &cvRects);
    bool overlaps(const QList<cv::Rect> &posRects, const cv::Rect &negRect, double overlap);
    float overlap(const cv::Rect &rect1, const cv::Rect &rect2);
    float overlap(const QRectF &rect1, const QRectF &rect2);
    QString rotatedRectToString(const cv::RotatedRect &rotatedRect);
    BR_EXPORT cv::RotatedRect rotateRectFromString(const QString &string, bool *ok);

    enum Axis { X = 0, Y = 1, Both = -1 };

    // Misc
    void group(QList<cv::Rect> &rects, QList<float> &confidences, float confidenceThreshold, int minNeighbors, float epsilon, bool useMax=false, QList<int> *maxIndices=NULL);
    void pad(const br::Template &src, br::Template &dst, bool padMat, const QMarginsF &padding, bool padPoints, bool padRects, int border=0, int value=0);
    void pad(const br::TemplateList &src, br::TemplateList &dst, bool padMat, const QMarginsF &padding, bool padPoints, bool padRects, int border=0, int value=0);
    QPointF rotatePoint(const QPointF &point, const cv::Mat &rotationMatrix);
    QList<QPointF> rotatePoints(const QList<QPointF> &points, const cv::Mat &rotationMatrix);
    QRectF rotateRect(const QRectF &rect, const cv::Mat &rotationMatrix);
    QList<QRectF> rotateRects(const QList<QRectF> &rects, const cv::Mat &rotationMatrix);
    void rotate(const br::Template &src, br::Template &dst, float degrees, bool rotateMat=true, bool rotatePoints=true, bool rotateRects=true, const QPointF &center = QPointF());
    void rotate(const br::TemplateList &src, br::TemplateList &dst, float degrees, bool rotateMat=true, bool rotatePoint=true, bool rotateRects=true, const QPointF &center = QPointF());
    QRectF flipRect(const cv::Mat &mat, const QRectF &rect, Axis axis);
    QList<QRectF> flipRects(const cv::Mat &mat, const QList<QRectF> &rects, Axis axis);
    void flip(const br::Template &src, br::Template &dst, Axis axis, bool flipMat=true, bool flipPoints=true, bool flipRects=true);
    void flip(const br::TemplateList &src, br::TemplateList &dst, Axis axis, bool flipMat=true, bool flipPoints=true, bool flipRects=true);

    int getFourcc();
}

BR_EXPORT QDebug operator<<(QDebug dbg, const cv::Mat &m);
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

// Templated OpenCV Mat::type creation.
template <typename Depth, int Channels>
class OpenCVType
{
    static int getDepth(Type2Type<uchar>)  { return CV_8U;  }
    static int getDepth(Type2Type<char>)   { return CV_8S;  }
    static int getDepth(Type2Type<ushort>) { return CV_16U; }
    static int getDepth(Type2Type<short>)  { return CV_16S; }
    static int getDepth(Type2Type<long>)   { return CV_32S; }
    static int getDepth(Type2Type<float>)  { return CV_32F; }
    static int getDepth(Type2Type<double>) { return CV_64F; }

public:
    static int make() { return CV_MAKETYPE((getDepth(Type2Type<Depth>())),(Channels)); }
};

#endif // OPENCVUTILS_OPENCVUTILS_H
