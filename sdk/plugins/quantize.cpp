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

#include <openbr_plugin.h>

#include "core/opencvutils.h"

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Approximate floats as uchar.
 * \author Josh Klontz \cite jklontz
 */
class QuantizeTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(float a READ get_a WRITE set_a RESET reset_a)
    Q_PROPERTY(float b READ get_b WRITE set_b RESET reset_b)
    BR_PROPERTY(float, a, 1)
    BR_PROPERTY(float, b, 0)

    void train(const TemplateList &data)
    {
        double minVal, maxVal;
        minMaxLoc(OpenCVUtils::toMat(data.data()), &minVal, &maxVal);
        a = 255.0/(maxVal-minVal);
        b = -a*minVal;
    }

    void project(const Template &src, Template &dst) const
    {
        src.m().convertTo(dst, CV_8U, a, b);
    }
};

BR_REGISTER(Transform, QuantizeTransform)

/*!
 * \ingroup transforms
 * \brief Approximate floats as signed bit.
 * \author Josh Klontz \cite jklontz
 */
class BinarizeTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        const Mat &m = src;
        if ((m.cols % 8 != 0) || (m.type() != CV_32FC1))
            qFatal("Expected CV_32FC1 matrix with a multiple of 8 columns.");
        Mat n(m.rows, m.cols/8, CV_8UC1);
        for (int i=0; i<m.rows; i++)
            for (int j=0; j<m.cols-7; j+=8)
                n.at<uchar>(i,j) = ((m.at<float>(i,j+0) > 0) << 0) +
                                   ((m.at<float>(i,j+1) > 0) << 1) +
                                   ((m.at<float>(i,j+2) > 0) << 2) +
                                   ((m.at<float>(i,j+3) > 0) << 3) +
                                   ((m.at<float>(i,j+4) > 0) << 4) +
                                   ((m.at<float>(i,j+5) > 0) << 5) +
                                   ((m.at<float>(i,j+6) > 0) << 6) +
                                   ((m.at<float>(i,j+7) > 0) << 7);
        dst = n;
    }
};

BR_REGISTER(Transform, BinarizeTransform)

/*!
 * \ingroup transforms
 * \brief Compress two uchar into one uchar.
 * \author Josh Klontz \cite jklontz
 */
class PackTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        const Mat &m = src;
        if ((m.cols % 2 != 0) || (m.type() != CV_8UC1))
            qFatal("Invalid template format.");

        Mat n(m.rows, m.cols/2, CV_8UC1);
        for (int i=0; i<m.rows; i++)
            for (int j=0; j<m.cols/2; j++)
                n.at<uchar>(i,j) = ((m.at<uchar>(i,2*j+0) >> 4) << 4) +
                                   ((m.at<uchar>(i,2*j+1) >> 4) << 0);
        dst = n;
    }
};

BR_REGISTER(Transform, PackTransform)

QVector<Mat> BayesianProductQuantizationLUTs;

/*!
 * \ingroup transforms
 * \brief A bayesian extension to product quantization \cite jegou11
 * \author Josh Klontz \cite jklontz
 */
class BayesianProductQuantizationTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(int n READ get_n WRITE set_n RESET reset_n STORED false)
    BR_PROPERTY(int, n, 3)

    int index;
    QList<Mat> centers;

public:
    BayesianProductQuantizationTransform()
    {
        index = BayesianProductQuantizationLUTs.size();
        BayesianProductQuantizationLUTs.append(Mat());
    }

private:
    void train(const TemplateList &src)
    {
        Mat data = OpenCVUtils::toMatByRow(src.data());
        if (data.cols % n != 0) qFatal("Expected dimensionality to be divisible by n.");

        Mat &lut = BayesianProductQuantizationLUTs[index];
        lut = Mat(data.cols/n, 256*256, CV_32FC1);

        for (int i=0; i<data.cols/n; i++) {
            Mat labels, center;
            kmeans(data.colRange(i*n,(i+1)*n), 256, labels, TermCriteria(TermCriteria::MAX_ITER, 10, 0), 3, KMEANS_PP_CENTERS, center);

            for (int j=0; j<256; j++)
                for (int k=0; k<256; k++)
                    lut.at<float>(i,j*256+k) = norm(center.row(k), center.row(k), NORM_L2);
            centers.append(center);
        }
    }

    void project(const Template &src, Template &dst) const
    {
        dst = Mat(1, src.m().cols/n, CV_8UC1);
        for (int i=0; i<src.m().cols/n; i++) {
            uchar bestIndex = -1;
            double bestDistance = std::numeric_limits<double>::max();
            Mat m = src.m().colRange(i*n, (i+1)*n);
            for (uchar i=0; i<256; i++) {
                double distance = norm(m, centers[index].row(i), NORM_L2);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestIndex = i;
                }
            }
            dst.m().at<uchar>(0,i) = bestIndex;
        }
    }

    void store(QDataStream &stream) const
    {
        stream << centers << BayesianProductQuantizationLUTs[index];
    }

    void load(QDataStream &stream)
    {
        stream >> centers >> BayesianProductQuantizationLUTs[index];
    }
};

BR_REGISTER(Transform, BayesianProductQuantizationTransform)

} // namespace br

#include "quantize.moc"
