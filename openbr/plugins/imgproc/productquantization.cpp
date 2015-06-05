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

#include <QtConcurrent>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/common.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

QVector<Mat> ProductQuantizationLUTs;

/*!
 * \ingroup distances
 * \brief Distance in a product quantized space
 * \br_paper Jegou, Herve, Matthijs Douze, and Cordelia Schmid.
 *           "Product quantization for nearest neighbor search."
 *           Pattern Analysis and Machine Intelligence, IEEE Transactions on 33.1 (2011): 117-128
 * \author Josh Klontz \cite jklontz
 */
class ProductQuantizationDistance : public UntrainableDistance
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
 * \ingroup transforms
 * \brief Product quantization
 * \br_paper Jegou, Herve, Matthijs Douze, and Cordelia Schmid.
 *           "Product quantization for nearest neighbor search."
 *           Pattern Analysis and Machine Intelligence, IEEE Transactions on 33.1 (2011): 117-128
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

#include "imgproc/productquantization.moc"
