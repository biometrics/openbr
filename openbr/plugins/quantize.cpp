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

#include <QFutureSynchronizer>
#include <QtConcurrentRun>
#include <openbr/openbr_plugin.h>

#include "openbr/core/opencvutils.h"

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

QVector<Mat> ProductQuantizationLUTs;

/*!
 * \ingroup distances
 * \brief Distance in a product quantized space \cite jegou11
 * \author Josh Klontz
 */
class ProductQuantizationDistance : public Distance
{
    Q_OBJECT
    Q_PROPERTY(bool bayesian READ get_bayesian WRITE set_bayesian RESET reset_bayesian STORED false)
    BR_PROPERTY(bool, bayesian, false)

    float compare(const Template &a, const Template &b) const
    {
        float distance = 0;
        for (int i=0; i<a.size(); i++) {
            const int elements = a[i].total();
            const uchar *aData = a[i].data;
            const uchar *bData = b[i].data;
            const float *lut = (const float*)ProductQuantizationLUTs[i].data;
            for (int j=0; j<elements; j++)
                 distance += lut[i*256*256 + aData[j]*256+bData[j]];
        }
        if (!bayesian) distance = -log(distance+1);
        return distance;
    }
};

BR_REGISTER(Distance, ProductQuantizationDistance)

/*!
 * \ingroup transforms
 * \brief Product quantization \cite jegou11
 * \author Josh Klontz \cite jklontz
 */
class ProductQuantizationTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(int n READ get_n WRITE set_n RESET reset_n STORED false)
    Q_PROPERTY(bool bayesian READ get_bayesian WRITE set_bayesian RESET reset_bayesian STORED false)
    BR_PROPERTY(int, n, 2)
    BR_PROPERTY(bool, bayesian, false)

    int index;
    QList<Mat> centers;

public:
    ProductQuantizationTransform()
    {
        index = ProductQuantizationLUTs.size();
        ProductQuantizationLUTs.append(Mat());
    }

private:
    static double likelihoodRatio(const QPair<int,int> &totals, const QList<int> &targets, const QList<int> &queries)
    {
        int positives = 1, negatives = 1; // Equal priors
        foreach (int t, targets)
            foreach (int q, queries)
                if (t == q) positives++;
                else        negatives++;
        return log((float(positives)/float(totals.first)) / (float(negatives)/float(totals.second)));
    }

    void _train(const Mat &data, const QPair<int,int> &totals, Mat &lut, int i, const QList<int> &templateLabels)
    {
        Mat labels, center;
        kmeans(data.colRange(i*n,(i+1)*n), 256, labels, TermCriteria(TermCriteria::MAX_ITER, 10, 0), 3, KMEANS_PP_CENTERS, center);
        QList<int> clusterLabels = OpenCVUtils::matrixToVector<int>(labels);

        QHash< int, QList<int> > clusters; // QHash<clusterLabel, QList<templateLabel>>
        for (int i=0; i<clusterLabels.size(); i++)
            clusters[clusterLabels[i]].append(templateLabels[i]);

        for (int j=0; j<256; j++)
            for (int k=0; k<256; k++)
                lut.at<float>(i,j*256+k) = bayesian ? likelihoodRatio(totals, clusters[j], clusters[k]) :
                                                      norm(center.row(j), center.row(k), NORM_L2);
        centers[i] = center;
    }

    void train(const TemplateList &src)
    {
        Mat data = OpenCVUtils::toMat(src.data());
        if (data.cols % n != 0) qFatal("Expected dimensionality to be divisible by n.");
        const QList<int> templateLabels = src.labels<int>();
        int totalPositives = 0, totalNegatives = 0;
        for (int i=0; i<templateLabels.size(); i++)
            for (int j=0; j<templateLabels.size(); j++)
                if (templateLabels[i] == templateLabels[j]) totalPositives++;
                else                                        totalNegatives++;
        QPair<int,int> totals(totalPositives, totalNegatives);

        Mat &lut = ProductQuantizationLUTs[index];
        lut = Mat(data.cols/n, 256*256, CV_32FC1);

        for (int i=0; i<lut.rows; i++)
            centers.append(Mat());

        QFutureSynchronizer<void> futures;
        for (int i=0; i<lut.rows; i++) {
            if (Globals->parallelism) futures.addFuture(QtConcurrent::run(this, &ProductQuantizationTransform::_train, data, totals, lut, i, templateLabels));
            else                                                                                               _train (data, totals, lut, i, templateLabels);
        }
        futures.waitForFinished();
    }

    void project(const Template &src, Template &dst) const
    {
        Mat m = src.m().reshape(1, 1);
        dst = Mat(1, m.cols/n, CV_8UC1);
        for (int i=0; i<dst.m().cols; i++) {
            int bestIndex = 0;
            double bestDistance = std::numeric_limits<double>::max();
            Mat m_i = m.colRange(i*n, (i+1)*n);
            for (int j=0; j<256; j++) {
                double distance = norm(m_i, centers[index].row(j), NORM_L2);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestIndex = j;
                }
            }
            dst.m().at<uchar>(0,i) = bestIndex;
        }
    }

    void store(QDataStream &stream) const
    {
        stream << centers << ProductQuantizationLUTs[index];
    }

    void load(QDataStream &stream)
    {
        stream >> centers >> ProductQuantizationLUTs[index];
    }
};

BR_REGISTER(Transform, ProductQuantizationTransform)

} // namespace br

#include "quantize.moc"
