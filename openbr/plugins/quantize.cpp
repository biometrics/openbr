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
#include "openbr_internal.h"

#include "openbr/core/common.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/qtutils.h"

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
 * \brief Approximate floats as uchar with different scalings for each dimension.
 * \author Josh Klontz \cite jklontz
 */
class HistEqQuantizationTransform : public Transform
{
    Q_OBJECT
    QVector<float> thresholds;

    static void computeThresholds(const Mat &data, float *thresholds)
    {
        QList<float> vals = OpenCVUtils::matrixToVector<float>(data);
        std::sort(vals.begin(), vals.end());
        for (int i=0; i<255; i++)
            thresholds[i] = vals[(i+1)*vals.size()/256];
        thresholds[255] = std::numeric_limits<float>::max();
    }

    void train(const TemplateList &src)
    {
        const Mat data = OpenCVUtils::toMat(src.data());
        thresholds = QVector<float>(256*data.cols);

        QFutureSynchronizer<void> futures;
        for (int i=0; i<data.cols; i++)
            futures.addFuture(QtConcurrent::run(&HistEqQuantizationTransform::computeThresholds, data.col(i), &thresholds.data()[i*256]));
        futures.waitForFinished();
    }

    void project(const Template &src, Template &dst) const
    {
        const QList<float> vals = OpenCVUtils::matrixToVector<float>(src);
        dst = Mat(1, vals.size(), CV_8UC1);
        for (int i=0; i<vals.size(); i++) {
            const float *t = &thresholds.data()[i*256];
            const float val = vals[i];
            uchar j = 0;
            while (val > t[j]) j++;
            dst.m().at<uchar>(0,i) = j;
        }
    }

    void store(QDataStream &stream) const
    {
        stream << thresholds;
    }

    void load(QDataStream &stream)
    {
        stream >> thresholds;
    }
};

BR_REGISTER(Transform, HistEqQuantizationTransform)

/*!
 * \ingroup distances
 * \brief Bayesian quantization distance
 * \author Josh Klontz \cite jklontz
 */
class BayesianQuantizationDistance : public Distance
{
    Q_OBJECT

    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED false)
    BR_PROPERTY(QString, inputVariable, "Label")

    QVector<float> loglikelihoods;

    static void computeLogLikelihood(const Mat &data, const QList<int> &labels, float *loglikelihood)
    {
        const QList<uchar> vals = OpenCVUtils::matrixToVector<uchar>(data);
        if (vals.size() != labels.size())
            qFatal("Logic error.");

        QVector<quint64> genuines(256, 0), impostors(256,0);
        for (int i=0; i<vals.size(); i++)
            for (int j=i+1; j<vals.size(); j++)
                if (labels[i] == labels[j]) genuines[abs(vals[i]-vals[j])]++;
                else                        impostors[abs(vals[i]-vals[j])]++;

        quint64 totalGenuines(0), totalImpostors(0);
        for (int i=0; i<256; i++) {
            totalGenuines += genuines[i];
            totalImpostors += impostors[i];
        }

        for (int i=0; i<256; i++)
            loglikelihood[i] = log((float(genuines[i]+1)/totalGenuines)/(float(impostors[i]+1)/totalImpostors));
    }

    void train(const TemplateList &src)
    {
        if ((src.first().size() > 1) || (src.first().m().type() != CV_8UC1))
            qFatal("Expected sigle matrix templates of type CV_8UC1!");

        const Mat data = OpenCVUtils::toMat(src.data());
        const QList<int> templateLabels = src.indexProperty(inputVariable);
        loglikelihoods = QVector<float>(data.cols*256, 0);

        QFutureSynchronizer<void> futures;
        for (int i=0; i<data.cols; i++)
            futures.addFuture(QtConcurrent::run(&BayesianQuantizationDistance::computeLogLikelihood, data.col(i), templateLabels, &loglikelihoods.data()[i*256]));
        futures.waitForFinished();
    }

    float compare(const cv::Mat &a, const cv::Mat &b) const
    {
        const uchar *aData = a.data;
        const uchar *bData = b.data;
        const int size = a.rows * a.cols;
        float likelihood = 0;
        for (int i=0; i<size; i++)
            likelihood += loglikelihoods[i*256+abs(aData[i]-bData[i])];
        return likelihood;
    }

    void store(QDataStream &stream) const
    {
        stream << loglikelihoods;
    }

    void load(QDataStream &stream)
    {
        stream >> loglikelihoods;
    }
};

BR_REGISTER(Distance, BayesianQuantizationDistance)

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
 * \author Josh Klontz \cite jklontz
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
            const int elements = a[i].total()-sizeof(quint16);
            uchar *aData = a[i].data;
            uchar *bData = b[i].data;
            quint16 index = *reinterpret_cast<quint16*>(aData);
            aData += sizeof(quint16);
            bData += sizeof(quint16);

            const float *lut = (const float*)ProductQuantizationLUTs[index].data;
            for (int j=0; j<elements; j++)
            {
                const int aj = aData[j];
                const int bj = bData[j];
                // http://stackoverflow.com/questions/4803180/mapping-elements-in-2d-upper-triangle-and-lower-triangle-to-linear-structure
                const int y = max(aj, bj);
                const int x = min(aj, bj);
                distance += lut[j*256*(256+1)/2 + x + (y+1)*y/2];
            }
        }
        if (!bayesian) distance = -log(distance+1);
        return distance;
    }
};

BR_REGISTER(Distance, ProductQuantizationDistance)

/*!
 * \ingroup distances
 * \brief Recurively computed distance in a product quantized space
 * \author Josh Klontz \cite jklontz
 */
class RecursiveProductQuantizationDistance : public Distance
{
    Q_OBJECT
    Q_PROPERTY(float t READ get_t WRITE set_t RESET reset_t STORED false)
    BR_PROPERTY(float, t, -std::numeric_limits<float>::max())

    float compare(const Template &a, const Template &b) const
    {
        return compareRecursive(a, b, 0, a.size(), 0);
    }

    float compareRecursive(const QList<cv::Mat> &a, const QList<cv::Mat> &b, int i, int size, float evidence) const
    {
        float similarity = 0;

        const int elements = a[i].total()-sizeof(quint16);
        uchar *aData = a[i].data;
        uchar *bData = b[i].data;
        quint16 index = *reinterpret_cast<quint16*>(aData);
        aData += sizeof(quint16);
        bData += sizeof(quint16);

        const float *lut = (const float*)ProductQuantizationLUTs[index].data;
        for (int j=0; j<elements; j++) {
            const int aj = aData[j];
            const int bj = bData[j];
            // http://stackoverflow.com/questions/4803180/mapping-elements-in-2d-upper-triangle-and-lower-triangle-to-linear-structure
            const int y = max(aj, bj);
            const int x = min(aj, bj);
            similarity += lut[j*256*(256+1)/2 + x + (y+1)*y/2];
        }

        evidence += similarity;
        const int subSize = (size-1)/4;
        if ((evidence < t) || (subSize == 0)) return similarity;
        return similarity
               + compareRecursive(a, b, i+1+0*subSize, subSize, evidence)
               + compareRecursive(a, b, i+1+1*subSize, subSize, evidence)
               + compareRecursive(a, b, i+1+2*subSize, subSize, evidence)
               + compareRecursive(a, b, i+1+3*subSize, subSize, evidence);
    }
};

BR_REGISTER(Distance, RecursiveProductQuantizationDistance)

/*!
 * \ingroup transforms
 * \brief Product quantization \cite jegou11
 * \author Josh Klontz \cite jklontz
 */
class ProductQuantizationTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(int n READ get_n WRITE set_n RESET reset_n STORED false)
    Q_PROPERTY(br::Distance *distance READ get_distance WRITE set_distance RESET reset_distance STORED false)
    Q_PROPERTY(bool bayesian READ get_bayesian WRITE set_bayesian RESET reset_bayesian STORED false)
    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED false)
    BR_PROPERTY(int, n, 2)
    BR_PROPERTY(br::Distance*, distance, Distance::make("L2", this))
    BR_PROPERTY(bool, bayesian, false)
    BR_PROPERTY(QString, inputVariable, "Label")

    quint16 index;
    QList<Mat> centers;

public:
    ProductQuantizationTransform()
    {
        if (ProductQuantizationLUTs.size() > std::numeric_limits<quint16>::max())
            qFatal("Out of LUT space!"); // Unlikely

        static QMutex mutex;
        QMutexLocker locker(&mutex);
        index = ProductQuantizationLUTs.size();
        ProductQuantizationLUTs.append(Mat());
    }

private:
//    static double denseKernelDensityBandwidth(const Mat &lut, const Mat &occurences)
//    {
//        double total = 0;
//        int n = 0;
//        const qint32 *occurencesData = (qint32*)occurences.data;
//        const float *lutData = (float*)lut.data;
//        for (int i=0; i<256; i++)
//            for (int j=i; j<256; j++) {
//                total += occurencesData[i*256+j] * lutData[i*256+j];
//                n += occurencesData[i*256+j];
//            }
//        const double mean = total/n;

//        double variance = 0;
//        for (int i=0; i<lut.rows; i++)
//            for (int j=i; j<lut.cols; j++)
//                variance += occurencesData[i*256+j] * pow(lutData[i*256+j]-mean, 2.0);

//        return pow(4 * pow(sqrt(variance/n), 5.0) / (3*n), 0.2);
//    }

//    static double denseKernelDensityEstimation(const Mat &lut, const Mat &occurences, const float x, const float h)
//    {
//        double y = 0;
//        int n = 0;
//        const qint32 *occurencesData = (qint32*)occurences.data;
//        const float *lutData = (float*)lut.data;
//        for (int i=0; i<256; i++)
//            for (int j=i; j<256; j++) {
//                const int n_ij = occurencesData[i*256+j];
//                if (n_ij > 0) {
//                    y += n_ij * exp(-pow((lutData[i*256+j]-x)/h,2)/2)/2.50662826737 /*sqrt(2*3.1415926353898)*/;
//                    n += n_ij;
//                }
//            }
//        return y / (n*h);
//    }

    void _train(const Mat &data, const QList<int> &labels, Mat *lut, Mat *center)
    {
        Mat clusterLabels;
        kmeans(data, 256, clusterLabels, TermCriteria(TermCriteria::MAX_ITER, 10, 0), 3, KMEANS_PP_CENTERS, *center);

        Mat fullLUT(1, 256*256, CV_32FC1);
        for (int i=0; i<256; i++)
            for (int j=0; j<256; j++)
                fullLUT.at<float>(0,i*256+j) = distance->compare(center->row(i), center->row(j));

        if (bayesian) {
            QList<int> indicies = OpenCVUtils::matrixToVector<int>(clusterLabels);
            QVector<float> genuineScores, impostorScores;
            genuineScores.reserve(indicies.size());
            impostorScores.reserve(indicies.size()*indicies.size()/2);
            for (int i=0; i<indicies.size(); i++)
                for (int j=i+1; j<indicies.size(); j++) {
                    const float score = fullLUT.at<float>(0, indicies[i]*256+indicies[j]);
                    if (labels[i] == labels[j]) genuineScores.append(score);
                    else                        impostorScores.append(score);
                }

            genuineScores = Common::Downsample(genuineScores, 256);
            impostorScores = Common::Downsample(impostorScores, 256);
            const double hGenuine = Common::KernelDensityBandwidth(genuineScores);
            const double hImpostor = Common::KernelDensityBandwidth(impostorScores);

            for (int i=0; i<256; i++)
                for (int j=i; j<256; j++) {
                    const float loglikelihood = log(Common::KernelDensityEstimation(genuineScores, fullLUT.at<float>(0,i*256+j), hGenuine) /
                                                    Common::KernelDensityEstimation(impostorScores, fullLUT.at<float>(0,i*256+j), hImpostor));
                    fullLUT.at<float>(0,i*256+j) = loglikelihood;
                    fullLUT.at<float>(0,j*256+i) = loglikelihood;
                }
        }

        // Compress LUT into one dimensional array
        int index = 0;
        for (int i=0; i<256; i++)
            for (int j=0; j<=i; j++) {
                lut->at<float>(0,index) = fullLUT.at<float>(0,i*256+j);
                index++;
            }
        if (index != lut->cols)
            qFatal("Logic error.");
    }

    int getStep(int cols) const
    {
        if (n > 0) return n;
        if (n == 0) return cols;
        return ceil(float(cols)/abs(n));
    }

    int getOffset(int cols) const
    {
        if (n >= 0) return 0;
        const int step = getStep(cols);
        return (step - cols%step) % step;
    }

    int getDims(int cols) const
    {
        const int step = getStep(cols);
        if (n >= 0) return cols/step;
        return ceil(float(cols)/step);
    }

    void train(const TemplateList &src)
    {
        Mat data = OpenCVUtils::toMat(src.data());
        const int step = getStep(data.cols);

        const QList<int> labels = src.indexProperty(inputVariable);

        Mat &lut = ProductQuantizationLUTs[index];
        lut = Mat(getDims(data.cols), 256*(256+1)/2, CV_32FC1);

        QList<Mat> subdata, subluts;
        const int offset = getOffset(data.cols);
        for (int i=0; i<lut.rows; i++) {
            centers.append(Mat());
            subdata.append(data.colRange(max(0, i*step-offset), (i+1)*step-offset));
            subluts.append(lut.row(i));
        }

        QFutureSynchronizer<void> futures;
        for (int i=0; i<lut.rows; i++) {
            if (Globals->parallelism) futures.addFuture(QtConcurrent::run(this, &ProductQuantizationTransform::_train, subdata[i], labels, &subluts[i], &centers[i]));
            else                                                                                               _train (subdata[i], labels, &subluts[i], &centers[i]);
        }
        futures.waitForFinished();
    }

    int getIndex(const Mat &m, const Mat &center) const
    {
        int bestIndex = 0;
        double bestDistance = std::numeric_limits<double>::max();
        for (int j=0; j<256; j++) {
            double distance = norm(m, center.row(j), NORM_L2);
            if (distance < bestDistance) {
                bestDistance = distance;
                bestIndex = j;
            }
        }
        return bestIndex;
    }

    void project(const Template &src, Template &dst) const
    {
        Mat m = src.m().reshape(1, 1);
        const int step = getStep(m.cols);
        const int offset = getOffset(m.cols);
        const int dims = getDims(m.cols);
        dst = Mat(1, sizeof(quint16)+dims, CV_8UC1);
        memcpy(dst.m().data, &index, sizeof(quint16));
        for (int i=0; i<dims; i++)
            dst.m().at<uchar>(0,sizeof(quint16)+i) = getIndex(m.colRange(max(0, i*step-offset), (i+1)*step-offset), centers[i]);
    }

    void store(QDataStream &stream) const
    {
        stream << index << centers << ProductQuantizationLUTs[index];
    }

    void load(QDataStream &stream)
    {
        stream >> index >> centers;
        while (ProductQuantizationLUTs.size() <= index)
            ProductQuantizationLUTs.append(Mat());
        stream >> ProductQuantizationLUTs[index];
    }
};

BR_REGISTER(Transform, ProductQuantizationTransform)

} // namespace br

#include "quantize.moc"
