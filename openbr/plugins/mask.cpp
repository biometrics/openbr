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

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include "openbr_internal.h"
#include "openbr/core/opencvutils.h"

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Applies an eliptical mask
 * \author Josh Klontz \cite jklontz
 */
class MaskTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        const Mat &m = src;
        Mat mask(m.size(), CV_8UC1);
        mask.setTo(1);
        const float SCALE = 1.1;
        ellipse(mask, RotatedRect(Point2f(m.cols/2, m.rows/2), Size2f(SCALE*m.cols, SCALE*m.rows), 0), 0, -1);
        dst = m.clone();
        dst.m().setTo(0, mask);
    }
};

BR_REGISTER(Transform, MaskTransform)

/*!
 * \ingroup transforms
 * \brief Masks image according to pixel change.
 * \author Josh Klontz \cite jklontz
 */
class GradientMaskTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int delta READ get_delta WRITE set_delta RESET reset_delta STORED false)
    BR_PROPERTY(int, delta, 1)

    void project(const Template &src, Template &dst) const
    {
        const Mat &m = src.m();
        if (m.type() != CV_8UC1) qFatal("Requires 8UC1 matrices.");
        Mat n = Mat(m.rows, m.cols, CV_8UC1);
        n.setTo(255);
        for (int i=0; i<m.rows; i++) {
            for (int j=0; j<m.cols; j++) {
                if ((i>0)        && (abs(m.at<quint8>(i-1,j)-m.at<quint8>(i,j)) > delta)) { n.at<quint8>(i,j) = 0; continue; }
                if ((j+1<m.cols) && (abs(m.at<quint8>(i,j+1)-m.at<quint8>(i,j)) > delta)) { n.at<quint8>(i,j) = 0; continue; }
                if ((i+1<m.rows) && (abs(m.at<quint8>(i+1,j)-m.at<quint8>(i,j)) > delta)) { n.at<quint8>(i,j) = 0; continue; }
                if ((j>0)        && (abs(m.at<quint8>(i,j-1)-m.at<quint8>(i,j)) > delta)) { n.at<quint8>(i,j) = 0; continue; }
            }
        }
        dst = n;
    }
};

BR_REGISTER(Transform, GradientMaskTransform)

/*!
 * \ingroup transforms
 * \brief http://worldofcameras.wordpress.com/tag/skin-detection-opencv/
 * \author Josh Klontz \cite jklontz
 */
class SkinMaskTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        Mat m;
        cvtColor(src, m, CV_BGR2YCrCb);
        std::vector<Mat> mv;
        split(m, mv);
        Mat mask = Mat(m.rows, m.cols, CV_8UC1);

        for (int i=0; i<m.rows; i++) {
            for (int j=0; j<m.cols; j++) {
                int Cr= mv[1].at<quint8>(i,j);
                int Cb =mv[2].at<quint8>(i,j);
                mask.at<quint8>(i, j) = (Cr>130 && Cr<170) && (Cb>70 && Cb<125) ? 255 : 0;
            }
        }

        dst = mask;
    }
};

BR_REGISTER(Transform, SkinMaskTransform)

/*!
 * \ingroup transforms
 * \brief Morphological operator
 * \author Josh Klontz \cite jklontz
 */
class MorphTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_ENUMS(Op)
    Q_PROPERTY(Op op READ get_op WRITE set_op RESET reset_op STORED false)
    Q_PROPERTY(int radius READ get_radius WRITE set_radius RESET reset_radius STORED false)

public:
    /*!< */
    enum Op { Erode = MORPH_ERODE,
              Dilate = MORPH_DILATE,
              Open = MORPH_OPEN,
              Close = MORPH_CLOSE,
              Gradient = MORPH_GRADIENT,
              TopHat = MORPH_TOPHAT,
              BlackHat = MORPH_BLACKHAT };

private:
    BR_PROPERTY(Op, op, Close)
    BR_PROPERTY(int, radius, 1)

    Mat kernel;

    void init()
    {
        Mat kernel = Mat(radius, radius, CV_8UC1);
        kernel.setTo(255);
    }

    void project(const Template &src, Template &dst) const
    {
        morphologyEx(src, dst, op, kernel);
    }
};

BR_REGISTER(Transform, MorphTransform)

/*!
 * \ingroup transforms
 * \brief Set the template's label to the area of the largest convex hull.
 * \author Josh Klontz \cite jklontz
 */
class LargestConvexAreaTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(QString outputVariable READ get_outputVariable WRITE set_outputVariable RESET reset_outputVariable STORED false)
    BR_PROPERTY(QString, outputVariable, "Label")

    void project(const Template &src, Template &dst) const
    {
        std::vector< std::vector<Point> > contours;
        findContours(src.m().clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        double maxArea = 0;
        foreach (const std::vector<Point> &contour, contours) {
            std::vector<Point> hull;
            convexHull(contour, hull);
            double area = contourArea(contour);
            double hullArea = contourArea(hull);
            if (area / hullArea > 0.98)
                maxArea = std::max(maxArea, area);
        }
        dst.file.set(outputVariable, maxArea);
    }
};

BR_REGISTER(Transform, LargestConvexAreaTransform)

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV's adaptive thresholding.
 * \author Scott Klum \cite sklum
 */
class AdaptiveThresholdTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_ENUMS(Method)
    Q_ENUMS(Type)
    Q_PROPERTY(int maxValue READ get_maxValue WRITE set_maxValue RESET reset_maxValue STORED false)
    Q_PROPERTY(Method method READ get_method WRITE set_method RESET reset_method STORED false)
    Q_PROPERTY(Type type READ get_type WRITE set_type RESET reset_type STORED false)
    Q_PROPERTY(int blockSize READ get_blockSize WRITE set_blockSize RESET reset_blockSize STORED false)
    Q_PROPERTY(int C READ get_C WRITE set_C RESET reset_C STORED false)

    public:
    enum Method { Mean = ADAPTIVE_THRESH_MEAN_C,
                  Gaussian = ADAPTIVE_THRESH_GAUSSIAN_C };

    enum Type { Binary = THRESH_BINARY,
                Binary_Inv = THRESH_BINARY_INV };

    BR_PROPERTY(int, maxValue, 255)
    BR_PROPERTY(Method, method, Mean)
    BR_PROPERTY(Type, type, Binary)
    BR_PROPERTY(int, blockSize, 3)
    BR_PROPERTY(int, C, 0)

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        Mat mask;
        adaptiveThreshold(src, mask, maxValue, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, blockSize, C);

        dst.file.set("Mask",QVariant::fromValue(mask));
    }
};
BR_REGISTER(Transform, AdaptiveThresholdTransform)

/*!
 * \ingroup transforms
 * \brief Samples pixels from a mask.
 * \author Scott Klum \cite sklum
 */
class SampleFromMaskTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        Mat mask = src.file.get<Mat>("Mask");
        const int count = countNonZero(mask);
        dst.m() = Mat(1,count,src.m().type());

        Mat masked;
        src.m().copyTo(masked, mask);

        Mat indices;
        findNonZero(masked,indices);

        for (int j=0; j<indices.total(); j++)
            dst.m().at<uchar>(0,j) = masked.at<uchar>(indices.at<Point>(j).y,indices.at<Point>(j).x);
    }
};

BR_REGISTER(Transform, SampleFromMaskTransform)

/*!
 * \ingroup transforms
 * \brief Applies a mask from the metadata.
 * \author Austin Blanton \cite imaus10
 */
class ApplyMaskTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(QString key READ get_key WRITE set_key RESET reset_key STORED false)
    BR_PROPERTY(QString, key, "Mask")

    void project(const Template &src, Template &dst) const
    {
        if (src.file.contains(key))
            src.m().copyTo(dst, src.file.get<Mat>(key));
        else
            dst = src;
    }
};

BR_REGISTER(Transform, ApplyMaskTransform)

} // namespace br

#include "mask.moc"
