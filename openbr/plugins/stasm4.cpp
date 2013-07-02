#include <stasm_lib.h>
#include <stasmcascadeclassifier.h>
#include <opencv2/opencv.hpp>
#include "openbr_internal.h"
#include "openbr/core/qtutils.h"
#include "openbr/core/opencvutils.h"
#include <QString>
#include <Eigen/SVD>

using namespace cv;

namespace br
{

class StasmResourceMaker : public ResourceMaker<StasmCascadeClassifier>
{
private:
    StasmCascadeClassifier *make() const
    {
        StasmCascadeClassifier *stasmCascade = new StasmCascadeClassifier();
        if (!stasmCascade->load(Globals->sdkPath.toStdString() + "/share/openbr/models/"))
            qFatal("Failed to load Stasm Cascade");
        return stasmCascade;
    }
};

/*!
 * \ingroup initializers
 * \brief Initialize Stasm
 * \author Scott Klum \cite sklum
 */
class StasmInitializer : public Initializer
{
    Q_OBJECT

    void initialize() const
    {
        Globals->abbreviations.insert("RectFromStasmEyes","RectFromPoints([29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],0.125,6.0)+Resize(44,164)");
        Globals->abbreviations.insert("RectFromStasmBrow","RectFromPoints([17, 18, 19, 20, 21, 22, 23, 24],0.15,6)+Resize(28,132)");
        Globals->abbreviations.insert("RectFromStasmNose","RectFromPoints([48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58],0.15,1.25)+Resize(60,60)");
        Globals->abbreviations.insert("RectFromStasmMouth","RectFromPoints([59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76],0.3,3.0)+Resize(28,68)");
    }
};

BR_REGISTER(Initializer, StasmInitializer)

/*!
 * \ingroup transforms
 * \brief Wraps STASM key point detector
 * \author Scott Klum \cite sklum
 */
class StasmTransform : public UntrainableTransform
{
    Q_OBJECT

    Resource<StasmCascadeClassifier> stasmCascadeResource;

    void init()
    {
        if (!stasm_init(qPrintable(Globals->sdkPath + "/share/openbr/models/stasm"), 0)) qFatal("Failed to initalize stasm.");
        stasmCascadeResource.setResourceMaker(new StasmResourceMaker());
    }

    void project(const Template &src, Template &dst) const
    {
        StasmCascadeClassifier *stasmCascade = stasmCascadeResource.acquire();

        int foundface;
        float landmarks[2 * stasm_NLANDMARKS];
        stasm_search_single(&foundface, landmarks, reinterpret_cast<const char*>(src.m().data), src.m().cols, src.m().rows, *stasmCascade, NULL, NULL);

        stasmCascadeResource.release(stasmCascade);

        if (!foundface) qWarning("No face found in %s", qPrintable(src.file.fileName()));
        else {
            for (int i = 0; i < stasm_NLANDMARKS; i++)
                dst.file.appendPoint(QPointF(landmarks[2 * i], landmarks[2 * i + 1]));
        }

        dst.m() = src.m();
    }
};

BR_REGISTER(Transform, StasmTransform)

#include <iostream>

/*!
 * \ingroup transforms
 * \brief Wraps STASM key point detector
 * \author Scott Klum \cite sklum
 */
class ProcrustesTransform : public Transform
{
    Q_OBJECT

    Q_PROPERTY(QString principalShapePath READ get_principalShapePath WRITE set_principalShapePath RESET reset_principalShapePath STORED false)
    BR_PROPERTY(QString, principalShapePath, QString())

    Eigen::MatrixXf meanShape;

    void train(const TemplateList &data)
    {
        QList< QList<cv::Point2f> > normalizedPoints;

        // Normalize all sets of points
        foreach (br::Template datum, data) {
            QList<cv::Point2f> points = OpenCVUtils::toPoints(datum.file.points());

            if (points.empty()) {
                continue;
            }

            cv::Scalar mean = cv::mean(points.toVector().toStdVector());
            for (int i = 0; i < points.size(); i++) {
                points[i].x -= mean[0];
                points[i].y -= mean[1];
            }

            float norm = cv::norm(points.toVector().toStdVector());
            for (int i = 0; i < points.size(); i++) {
                points[i].x /= norm;
                points[i].y /= norm;
            }

            normalizedPoints.append(points);
        }

        // Determine mean shape
        Eigen::MatrixXf shapeTest(normalizedPoints[0].size(), 2);

        cv::Mat shapeBuffer(normalizedPoints[0].size(), 2, CV_32F);

        for (int i = 0; i < normalizedPoints[0].size(); i++) {

            double x = 0;
            double y = 0;

            for (int j = 0; j < normalizedPoints.size(); j++) {
                x += normalizedPoints[j][i].x;
                y += normalizedPoints[j][i].y;
            }

            x /= (double)normalizedPoints.size();
            y /= (double)normalizedPoints.size();

            shapeBuffer.at<float>(i,0) = x;
            shapeBuffer.at<float>(i,1) = y;

            shapeTest(i,0) = x;
            shapeTest(i,1) = y;
        }

        meanShape = shapeTest;
    }

    void project(const Template &src, Template &dst) const
    {
        QList<QPointF> points = src.file.points();

        cv::Scalar mean = cv::mean(OpenCVUtils::toPoints(points).toVector().toStdVector());

        for (int i = 0; i < points.size(); i++) points[i] -= QPointF(mean[0],mean[1]);

        float norm = cv::norm(OpenCVUtils::toPoints(points).toVector().toStdVector());

        Eigen::MatrixXf srcPoints(points.size(), 2);
        for (int i = 0; i < points.size(); i++) {
            srcPoints(i,0) = points[i].x()/norm;
            srcPoints(i,1) = points[i].y()/norm;
        }

        Eigen::JacobiSVD<Eigen::MatrixXf> svd(srcPoints.transpose()*meanShape, Eigen::ComputeThinU | Eigen::ComputeThinV);

        Eigen::MatrixXf R = svd.matrixU()*svd.matrixV().transpose();

        std::cout << R(1,0) << std::endl;
        // Determine transformation matrix

        // Apply transformation matrix
        //dst.file.setPoints(meanShape);*/
        dst.m() = src.m();
    }

};

BR_REGISTER(Transform, ProcrustesTransform)

} // namespace br

#include "stasm4.moc"
