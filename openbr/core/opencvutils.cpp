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

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <openbr/openbr_plugin.h>

#include "opencvutils.h"
#include "qtutils.h"
#include "common.h"

#include <QTemporaryFile>

using namespace cv;
using namespace std;

int OpenCVUtils::getFourcc()
{
    int fourcc = CV_FOURCC('x','2','6','4');
    QVariant recovered_variant = br::Globals->property("fourcc");

    if (!recovered_variant.isNull()) {
        QString recovered_string = recovered_variant.toString();
        if (recovered_string.length() == 4) {
            fourcc = CV_FOURCC(recovered_string[0].toLatin1(),
                               recovered_string[1].toLatin1(),
                               recovered_string[2].toLatin1(),
                               recovered_string[3].toLatin1());
        }
        else if (recovered_string.compare("-1")) fourcc = -1;
    }
    return fourcc;
}

void OpenCVUtils::saveImage(const Mat &src, const QString &file)
{
    if (file.isEmpty()) return;

    if (!src.data) {
        qWarning("OpenCVUtils::saveImage null image.");
        return;
    }

    QtUtils::touchDir(QFileInfo(file).dir());

    Mat draw;
    cvtUChar(src, draw);
    bool success = imwrite(file.toStdString(), draw); if (!success) qFatal("Failed to save %s", qPrintable(file));
}

void OpenCVUtils::showImage(const Mat &src, const QString &window, bool waitKey)
{
    if (!src.data) {
        qWarning("OpenCVUtils::showImage null image.");
        return;
    }

    Mat draw;
    cvtUChar(src, draw);
    imshow(window.toStdString(), draw);
    cv::waitKey(waitKey ? -1 : 1);
}

void OpenCVUtils::cvtGray(const Mat &src, Mat &dst)
{
    if      (src.channels() == 3) cvtColor(src, dst, CV_BGR2GRAY);
    else if (src.channels() == 1) dst = src;
    else                          qFatal("Invalid channel count");
}

void OpenCVUtils::cvtUChar(const Mat &src, Mat &dst)
{
    if (src.depth() == CV_8U) {
        dst = src;
        return;
    }

    double globalMin = std::numeric_limits<double>::max();
    double globalMax = -std::numeric_limits<double>::max();

    vector<Mat> mv;
    split(src, mv);
    for (size_t i=0; i<mv.size(); i++) {
        double min, max;
        minMaxLoc(mv[i], &min, &max);
        globalMin = std::min(globalMin, min);
        globalMax = std::max(globalMax, max);
    }
    assert(globalMax >= globalMin);

    double range = globalMax - globalMin;
    if (range != 0) {
        double scale = 255 / range;
        convertScaleAbs(src, dst, scale, -(globalMin * scale));
    } else {
        // Monochromatic
        dst = Mat(src.size(), CV_8UC1, Scalar((globalMin+globalMax)/2));
    }
}

Mat OpenCVUtils::toMat(const QList<float> &src, int rows)
{
    if (rows == -1) rows = src.size();
    int columns = src.isEmpty() ? 0 : src.size() / rows;
    if (rows*columns != src.size()) qFatal("Invalid matrix size.");
    Mat dst(rows, columns, CV_32FC1);
    for (int i=0; i<src.size(); i++)
        dst.at<float>(i/columns,i%columns) = src[i];
    return dst;
}

Mat OpenCVUtils::pointsToMatrix(const QList<QPointF> &qPoints)
{
    QList<float> points;
    foreach(const QPointF &point, qPoints) {
        points.append(point.x());
        points.append(point.y());
    }

    return toMat(points);
}

Mat OpenCVUtils::toMat(const QList<QList<float> > &srcs, int rows)
{
    QList<float> flat;
    foreach (const QList<float> &src, srcs)
        flat.append(src);
    return toMat(flat, rows);
}

Mat OpenCVUtils::toMat(const QList<int> &src, int rows)
{
    if (rows == -1) rows = src.size();
    int columns = src.isEmpty() ? 0 : src.size() / rows;
    if (rows*columns != src.size()) qFatal("Invalid matrix size.");
    Mat dst(rows, columns, CV_32FC1);
    for (int i=0; i<src.size(); i++)
        dst.at<float>(i/columns,i%columns) = src[i];
    return dst;
}

Mat OpenCVUtils::toMat(const QList<Mat> &src)
{
    if (src.isEmpty()) return Mat();

    int rows = src.size();
    size_t total = src.first().total();
    int type = src.first().type();
    Mat dst(rows, total, type);

    for (int i=0; i<rows; i++) {
        const Mat &m = src[i];
        if ((m.total() != total) || (m.type() != type) || !m.isContinuous())
            qFatal("Invalid matrix.");
        memcpy(dst.ptr(i), m.ptr(), total * src.first().elemSize());
    }
    return dst;
}

Mat OpenCVUtils::toMatByRow(const QList<Mat> &src)
{
    if (src.isEmpty()) return Mat();

    int rows = 0; foreach (const Mat &m, src) rows += m.rows;
    int cols = src.first().cols;
    if (cols == 0) qFatal("Columnless matrix!");
    int type = src.first().type();
    Mat dst(rows, cols, type);

    int row = 0;
    foreach (const Mat &m, src) {
        if ((m.cols != cols) || (m.type() != type) || (!m.isContinuous()))
            qFatal("Invalid matrix.");
        memcpy(dst.ptr(row), m.ptr(), m.rows*m.cols*m.elemSize());
        row += m.rows;
    }
    return dst;
}

QString OpenCVUtils::depthToString(const Mat &m)
{
    switch (m.depth()) {
      case CV_8U:  return "8U";
      case CV_8S:  return "8S";
      case CV_16U: return "16U";
      case CV_16S: return "16S";
      case CV_32S: return "32S";
      case CV_32F: return "32F";
      case CV_64F: return "64F";
      default:     qFatal("Unknown matrix depth!");
    }
    return "?";
}

QString OpenCVUtils::typeToString(const cv::Mat &m)
{
    return depthToString(m) + "C" + QString::number(m.channels());
}

QString OpenCVUtils::elemToString(const Mat &m, int r, int c)
{
    assert(m.channels() == 1);
    switch (m.depth()) {
      case CV_8U:  return QString::number(m.at<quint8>(r,c));
      case CV_8S:  return QString::number(m.at<qint8>(r,c));
      case CV_16U: return QString::number(m.at<quint16>(r,c));
      case CV_16S: return QString::number(m.at<qint16>(r,c));
      case CV_32S: return QString::number(m.at<qint32>(r,c));
      case CV_32F: return QString::number(m.at<float>(r,c));
      case CV_64F: return QString::number(m.at<double>(r,c));
      default:     qFatal("Unknown matrix depth");
    }
    return "?";
}

QString OpenCVUtils::matrixToString(const Mat &m)
{
    QString result;
    vector<Mat> mv;
    split(m, mv);
    if (m.rows > 1) result += "{ ";
    for (int r=0; r<m.rows; r++) {
        if ((m.rows > 1) && (r > 0)) result += "  ";
        if (m.cols > 1) result += "[";
        for (int c=0; c<m.cols; c++) {
            if (mv.size() > 1) result += "(";
            for (unsigned int i=0; i<mv.size()-1; i++)
                result += OpenCVUtils::elemToString(mv[i], r, c) + ", ";
            result += OpenCVUtils::elemToString(mv[mv.size()-1], r, c);
            if (mv.size() > 1) result += ")";
            if (c < m.cols - 1) result += ", ";
        }
        if (m.cols > 1) result += "]";
        if (r < m.rows-1) result += "\n";
    }
    if (m.rows > 1) result += " }";
    return result;
}

QStringList OpenCVUtils::matrixToStringList(const Mat &m)
{
    QStringList results;
    vector<Mat> mv;
    split(m, mv);
    foreach (const Mat &mc, mv)
        for (int i=0; i<mc.rows; i++)
            for (int j=0; j<mc.cols; j++)
                results.append(elemToString(mc, i, j));
    return results;
}

void OpenCVUtils::storeModel(const CvStatModel &model, QDataStream &stream)
{
    // Create local file
    QTemporaryFile tempFile;
    tempFile.open();
    tempFile.close();

    // Save MLP to local file
    model.save(qPrintable(tempFile.fileName()));

    // Copy local file contents to stream
    tempFile.open();
    QByteArray data = tempFile.readAll();
    tempFile.close();
    stream << data;
}

void OpenCVUtils::storeModel(const cv::Algorithm &model, QDataStream &stream)
{
    // Create local file
    QTemporaryFile tempFile;
    tempFile.open();
    tempFile.close();

    // Save MLP to local file
    cv::FileStorage fs(tempFile.fileName().toStdString(), cv::FileStorage::WRITE);
    model.write(fs);
    fs.release();

    // Copy local file contents to stream
    tempFile.open();
    QByteArray data = tempFile.readAll();
    tempFile.close();
    stream << data;
}

void OpenCVUtils::loadModel(CvStatModel &model, QDataStream &stream)
{
    // Copy file contents from stream
    QByteArray data;
    stream >> data;

    // This code for reading a file from memory inspired by CvStatModel::load implementation
    CvFileStorage *fs = cvOpenFileStorage(data.constData(), 0, CV_STORAGE_READ | CV_STORAGE_MEMORY);
    model.read(fs, (CvFileNode*) cvGetSeqElem(cvGetRootFileNode(fs)->data.seq, 0));
    cvReleaseFileStorage(&fs);
}

void OpenCVUtils::loadModel(cv::Algorithm &model, QDataStream &stream)
{
    // Copy local file contents from stream
    QByteArray data;
    stream >> data;

    // Create local file
    QTemporaryFile tempFile(QDir::tempPath()+"/model");
    tempFile.open();
    tempFile.write(data);
    tempFile.close();

    // Load MLP from local file
    cv::FileStorage fs(tempFile.fileName().toStdString(), cv::FileStorage::READ);
    model.read(fs[""]);
}

Point2f OpenCVUtils::toPoint(const QPointF &qPoint)
{
    return Point2f(qPoint.x(), qPoint.y());
}

QPointF OpenCVUtils::fromPoint(const Point2f &cvPoint)
{
    return QPointF(cvPoint.x, cvPoint.y);
}

QList<Point2f> OpenCVUtils::toPoints(const QList<QPointF> &qPoints)
{
    QList<Point2f> cvPoints; cvPoints.reserve(qPoints.size());
    foreach (const QPointF &qPoint, qPoints)
        cvPoints.append(toPoint(qPoint));
    return cvPoints;
}

QList<QPointF> OpenCVUtils::fromPoints(const QList<Point2f> &cvPoints)
{
    QList<QPointF> qPoints; qPoints.reserve(cvPoints.size());
    foreach (const Point2f &cvPoint, cvPoints)
        qPoints.append(fromPoint(cvPoint));
    return qPoints;
}

Rect OpenCVUtils::toRect(const QRectF &qRect)
{
    return Rect(qRect.x(), qRect.y(), qRect.width(), qRect.height());
}

RotatedRect OpenCVUtils::toRotatedRect(const QRectF &qRect, float angle)
{
    return RotatedRect(toPoint(qRect.center()), Size(qRect.width(), qRect.height()), angle);
}

QRectF OpenCVUtils::fromRect(const Rect &cvRect)
{
    return QRectF(cvRect.x, cvRect.y, cvRect.width, cvRect.height);
}

QList<Rect> OpenCVUtils::toRects(const QList<QRectF> &qRects)
{
    QList<Rect> cvRects; cvRects.reserve(qRects.size());
    foreach (const QRectF &qRect, qRects)
        cvRects.append(toRect(qRect));
    return cvRects;
}

QList<QRectF> OpenCVUtils::fromRects(const QList<Rect> &cvRects)
{
    QList<QRectF> qRects; qRects.reserve(cvRects.size());
    foreach (const Rect &cvRect, cvRects)
        qRects.append(fromRect(cvRect));
    return qRects;
}

float OpenCVUtils::overlap(const Rect &rect1, const Rect &rect2) {
    float left = max(rect1.x, rect2.x);
    float top = max(rect1.y, rect2.y);
    float right = min(rect1.x + rect1.width, rect2.x + rect2.width);
    float bottom = min(rect1.y + rect1.height, rect2.y + rect2.height);

    float overlap = (right - left + 1) * (top - bottom + 1) / max(rect1.width * rect1.height, rect2.width * rect2.height);
    if (overlap < 0)
        return 0;
    return overlap;
}

float OpenCVUtils::overlap(const QRectF &rect1, const QRectF &rect2) {
    float left = max(rect1.x(), rect2.x());
    float top = max(rect1.y(), rect2.y());
    float right = min(rect1.x() + rect1.width(), rect2.x() + rect2.width());
    float bottom = min(rect1.y() + rect1.height(), rect2.y() + rect2.height());

    float overlap = (right - left + 1) * (top - bottom + 1) / max(rect1.width() * rect1.height(), rect2.width() * rect2.height());
    if (overlap < 0)
        return 0;
    return overlap;
}

QString OpenCVUtils::rotatedRectToString(const RotatedRect &rotatedRect)
{
    return QString("RotatedRect(%1,%2,%3,%4,%5)").arg(QString::number(rotatedRect.center.x),
                                                      QString::number(rotatedRect.center.y),
                                                      QString::number(rotatedRect.size.width),
                                                      QString::number(rotatedRect.size.height),
                                                      QString::number(rotatedRect.angle));
}

cv::RotatedRect OpenCVUtils::rotateRectFromString(const QString &string, bool *ok)
{
    if (!string.startsWith("RotatedRect(") || !string.endsWith(")")) {
        *ok = false;
        return cv::RotatedRect();
    }

    const QStringList words = string.mid(12, string.size() - 13).split(",");
    if (words.size() != 5) {
        *ok = false;
        return cv::RotatedRect();
    }

    cv::RotatedRect result;
    result.center.x = words[0].toFloat(ok);
    if (!ok) return cv::RotatedRect();
    result.center.y = words[1].toFloat(ok);
    if (!ok) return cv::RotatedRect();
    result.size.width = words[2].toFloat(ok);
    if (!ok) return cv::RotatedRect();
    result.size.height = words[3].toFloat(ok);
    if (!ok) return cv::RotatedRect();
    result.angle = words[4].toFloat(ok);
    if (!ok) return cv::RotatedRect();

    *ok = true;
    return result;
}

bool OpenCVUtils::overlaps(const QList<Rect> &posRects, const Rect &negRect, double overlap)
{
    foreach (const Rect &posRect, posRects) {
        Rect intersect = negRect & posRect;
        if (intersect.area() > overlap*posRect.area())
            return true;
    }
    return false;
}

// class for grouping object candidates, detected by Cascade Classifier, HOG etc.
// instance of the class is to be passed to cv::partition (see cxoperations.hpp)
class SimilarRects
{
public:
    SimilarRects(double _eps) : eps(_eps) {}
    inline bool operator()(const Rect& r1, const Rect& r2) const
    {
        double delta = eps*(std::min(r1.width, r2.width) + std::min(r1.height, r2.height))*0.5;
        return std::abs(r1.x - r2.x) <= delta &&
            std::abs(r1.y - r2.y) <= delta &&
            std::abs(r1.x + r1.width - r2.x - r2.width) <= delta &&
            std::abs(r1.y + r1.height - r2.y - r2.height) <= delta;
    }
    double eps;
};

// TODO: Make sure case where no confidences are inputted works.
void OpenCVUtils::group(QList<Rect> &rects, QList<float> &confidences, float confidenceThreshold, int minNeighbors, float epsilon, bool useMax, QList<int> *maxIndices)
{
    if (rects.isEmpty())
        return;

    vector<int> labels;
    int nClasses = cv::partition(rects.toVector().toStdVector(), labels, SimilarRects(epsilon));

    // Rect for each class (class meaning identity assigned by partition)
    vector<Rect> rrects(nClasses);

    // Total number of rects in each class
    vector<int> neighbors(nClasses, -1);
    vector<float> classConfidence(nClasses, useMax ? -std::numeric_limits<float>::max() : 0);
    vector<int> classMax(nClasses, 0);

    for (size_t i = 0; i < labels.size(); i++)
    {
        int cls = labels[i];
        if (useMax) {
            if (confidences[i] > classConfidence[cls]) {
                classConfidence[cls] = confidences[i];
                classMax[cls] = i;
                rrects[cls].x = rects[i].x;
                rrects[cls].y = rects[i].y;
                rrects[cls].width = rects[i].width;
                rrects[cls].height = rects[i].height;
                neighbors[cls] = 0;
            }
        } else {
            classConfidence[cls] += confidences[i];
            rrects[cls].x += rects[i].x;
            rrects[cls].y += rects[i].y;
            rrects[cls].width += rects[i].width;
            rrects[cls].height += rects[i].height;
            neighbors[cls]++;
        }
    }

    // Find average rectangle for all classes
    for (int i = 0; i < nClasses; i++)
    {
        if (neighbors[i] > 0) {
            Rect r = rrects[i];
            float s = 1.f/(neighbors[i]+1);
            rrects[i] = Rect(saturate_cast<int>(r.x*s),
                 saturate_cast<int>(r.y*s),
                 saturate_cast<int>(r.width*s),
                 saturate_cast<int>(r.height*s));
        }
    }

    rects.clear();
    confidences.clear();

    // Aggregate by comparing average rectangles against other average rectangles
    for (int i = 0; i < nClasses; i++)
    {
        // Average rectangle
        const Rect r1 = rrects[i];

        // Used to eliminate rectangles with few neighbors in the case of no weights
        const float w1 = classConfidence[i];

        // Eliminate rectangle if it doesn't meet confidence criteria
        if (w1 < confidenceThreshold)
            continue;

        const int n1 = neighbors[i];
        if (n1 < minNeighbors)
            continue;

        // filter out small face rectangles inside large rectangles
        int j;
        for (j = 0; j < nClasses; j++)
        {
            const int n2 = neighbors[j];
            if (j == i || n2 < minNeighbors)
                continue;

            const Rect r2 = rrects[j];

            const int dx = saturate_cast<int>(r2.width * epsilon);
            const int dy = saturate_cast<int>(r2.height * epsilon);

            const float w2 = classConfidence[j];

            if(r1.x >= r2.x - dx &&
               r1.y >= r2.y - dy &&
               r1.x + r1.width <= r2.x + r2.width + dx &&
               r1.y + r1.height <= r2.y + r2.height + dy &&
               (w2 > w1) &&
               (n2 > n1))
               break;
        }

        if( j == nClasses )
        {
            rects.append(r1);
            confidences.append(w1);
            if (maxIndices)
                maxIndices->append(classMax[i]);
        }
    }
}

void OpenCVUtils::pad(const br::Template &src, br::Template &dst, bool padMat, const QMarginsF &padding, bool padPoints, bool padRects, int border, int value)
{
    // Padding is expected to be top, bottom, left, right
    if (padMat) {
        copyMakeBorder(src, dst, padding.top(), padding.bottom(), padding.left(), padding.right(), border, Scalar(value));
        dst.file = src.file;
    } else
        dst = src;

    if (padPoints) {
        QList<QPointF> points = src.file.points();
        QList<QPointF> paddedPoints;
        for (int i=0; i<points.size(); i++)
            paddedPoints.append(points[i] += QPointF(padding.left(),padding.top()));
        dst.file.setPoints(paddedPoints);
    }

    if (padRects) {
        QList<QRectF> rects = src.file.rects();
        QList<QRectF> paddedRects;
        for (int i=0; i<rects.size(); i++)
            paddedRects.append(rects[i].translated(QPointF(padding.left(),padding.top())));
        dst.file.setRects(paddedRects);
    }
}

void OpenCVUtils::pad(const br::TemplateList &src, br::TemplateList &dst, bool padMat, const QMarginsF &padding, bool padPoints, bool padRects, int border, int value)
{
    for (int i=0; i<src.size(); i++) {
        br::Template t;
        pad(src[i], t, padMat, padding, padPoints, padRects, border, value);
        dst.append(t);
    }
}

QPointF OpenCVUtils::rotatePoint(const QPointF &point, const Mat &rotationMatrix)
{
    return QPointF(point.x() * rotationMatrix.at<double>(0,0) +
                   point.y() * rotationMatrix.at<double>(0,1) +
                   1         * rotationMatrix.at<double>(0,2),
                   point.x() * rotationMatrix.at<double>(1,0) +
                   point.y() * rotationMatrix.at<double>(1,1) +
                   1         * rotationMatrix.at<double>(1,2));
}

QList<QPointF> OpenCVUtils::rotatePoints(const QList<QPointF> &points, const Mat &rotationMatrix)
{
    QList<QPointF> rotatedPoints;
    foreach (const QPointF &point, points)
        rotatedPoints.append(rotatePoint(point, rotationMatrix));
    return rotatedPoints;
}

QRectF OpenCVUtils::rotateRect(const QRectF &rect, const Mat &rotationMatrix)
{
    const QPointF center = OpenCVUtils::rotatePoint(rect.center(), rotationMatrix);
    return QRectF(center.x() - rect.width() / 2,
                  center.y() - rect.height() / 2,
                  rect.width(),
                  rect.height());
}

QList<QRectF> OpenCVUtils::rotateRects(const QList<QRectF> &rects, const Mat &rotationMatrix)
{
    QList<QRectF> rotatedRects;
    foreach (const QRectF &rect, rects)
        rotatedRects.append(rotateRect(rect, rotationMatrix));
    return rotatedRects;
}

void OpenCVUtils::rotate(const br::Template &src, br::Template &dst, float degrees, bool rotateMat, bool rotatePoints, bool rotateRects, const QPointF &center)
{
    const Mat rotMatrix = getRotationMatrix2D(center.isNull() ? Point2f(src.m().cols / 2, src.m().rows / 2) : toPoint(center), degrees, 1.0);

    if (rotateMat) {
        warpAffine(src, dst, rotMatrix, Size(src.m().cols, src.m().rows), INTER_AREA, BORDER_REPLICATE);
        dst.file = src.file;
    } else
        dst = src;

    if (rotatePoints)
        dst.file.setPoints(OpenCVUtils::rotatePoints(src.file.points(), rotMatrix));

    if (rotateRects)
        dst.file.setRects(OpenCVUtils::rotateRects(src.file.rects(), rotMatrix));
}

void OpenCVUtils::rotate(const br::TemplateList &src, br::TemplateList &dst, float degrees, bool rotateMat, bool rotatePoints, bool rotateRects, const QPointF &center)
{
    for (int i=0; i<src.size(); i++) {
        br::Template t;
        rotate(src[i], t, degrees, rotateMat, rotatePoints, rotateRects, center);
        dst.append(t);
    }
}

QRectF OpenCVUtils::flipRect(const cv::Mat &mat, const QRectF &rect, Axis axis)
{
    QRectF flippedRect;
    if (axis == X)
        flippedRect = QRectF(rect.x(),
                             mat.rows-rect.bottom(),
                             rect.width(),
                             rect.height());
    else if (axis == Y)
        flippedRect = QRectF(mat.cols-rect.right(),
                             rect.y(),
                             rect.width(),
                             rect.height());
    else
        flippedRect = QRectF(mat.cols-rect.right(),
                             mat.rows-rect.bottom(),
                             rect.width(),
                             rect.height());
    return flippedRect;
}

QList<QRectF> OpenCVUtils::flipRects(const cv::Mat &mat, const QList<QRectF> &rects, Axis axis)
{
    QList<QRectF> flippedRects;
    foreach(const QRectF &rect, rects)
        flippedRects.append(flipRect(mat, rect, axis));
    return flippedRects;
}

void OpenCVUtils::flip(const br::Template &src, br::Template &dst, Axis axis, bool flipMat, bool flipPoints, bool flipRects)
{
    if (flipMat) {
        cv::flip(src, dst, axis);
        dst.file = src.file;
    } else
        dst = src;

    if (flipPoints) {
        QList<QPointF> flippedPoints;
        foreach(const QPointF &point, src.file.points()) {
            // Check for missing data using the QPointF(-1,-1) convention
            if (point != QPointF(-1,-1)) {
                if (axis == X)
                    flippedPoints.append(QPointF(point.x(),src.m().rows-point.y()));
                else if (axis == Y)
                    flippedPoints.append(QPointF(src.m().cols-point.x(),point.y()));
                else
                    flippedPoints.append(QPointF(src.m().cols-point.x(),src.m().rows-point.y()));
            }
        }
        dst.file.setPoints(flippedPoints);
    }

    if (flipRects)
        dst.file.setRects(OpenCVUtils::flipRects(src, src.file.rects(), axis));
}

void OpenCVUtils::flip(const br::TemplateList &src, br::TemplateList &dst, Axis axis, bool flipMat, bool flipPoints, bool flipRects)
{
    for (int i=0; i<src.size(); i++) {
        br::Template t;
        flip(src[i], t, axis, flipMat, flipPoints, flipRects);
        dst.append(t);
    }
}

QDataStream &operator<<(QDataStream &stream, const Mat &m)
{
    // Write header
    int rows = m.rows;
    int cols = m.cols;
    int type = m.type();
    stream << rows << cols << type;

    // Write data
    int len = rows * cols * m.elemSize();
    stream << len;
    if (len > 0) {
        if (!m.isContinuous()) qFatal("Can't serialize non-continuous matrices.");
        int written = stream.writeRawData((const char*)m.data, len);
        if (written != len) qFatal("Mat serialization failure, expected: %d bytes, wrote: %d bytes.", len, written);
    }
    return stream;
}

QDataStream &operator>>(QDataStream &stream, Mat &m)
{
    // Read header
    int rows, cols, type;
    stream >> rows >> cols >> type;
    m.create(rows, cols, type);

    int len;
    stream >> len;
    char *data = (char*) m.data;

    // In certain circumstances, like reading from stdin or sockets, we may not
    // be given all the data we need at once because it isn't available yet.
    // So we loop until it we get it.
    while (len > 0) {
        const int read = stream.readRawData(data, len);
        if (read == -1) qFatal("Mat deserialization failure, exptected %d more bytes.", len);
        data += read;
        len -= read;
    }
    return stream;
}

QDebug operator<<(QDebug dbg, const Mat &m)
{
    dbg.nospace() << OpenCVUtils::matrixToString(m);
    return dbg.space();
}

QDebug operator<<(QDebug dbg, const Point &p)
{
    dbg.nospace() << "(" << p.x << ", " << p.y << ")";
    return dbg.space();
}

QDebug operator<<(QDebug dbg, const Rect &r)
{
    dbg.nospace() << "(" << r.x << ", " << r.y << "," << r.width << "," << r.height << ")";
    return dbg.space();
}

QDataStream &operator<<(QDataStream &stream, const Rect &r)
{
    return stream << r.x << r.y << r.width << r.height;
}

QDataStream &operator>>(QDataStream &stream, Rect &r)
{
    return stream >> r.x >> r.y >> r.width >> r.height;
}

QDataStream &operator<<(QDataStream &stream, const Size &s)
{
    return stream << s.width << s.height;
}

QDataStream &operator>>(QDataStream &stream, Size &s)
{
    return stream >> s.width >> s.height;
}
