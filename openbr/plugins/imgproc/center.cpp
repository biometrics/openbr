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

/*!
 * \ingroup transforms
 * \brief Normalize each dimension based on training data.
 * \author Josh Klontz \cite jklontz
 */
class CenterTransform : public Transform
{
    Q_OBJECT
    Q_ENUMS(Method)
    Q_PROPERTY(Method method READ get_method WRITE set_method RESET reset_method STORED false)
    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED false)

public:
    /*!< */
    enum Method { Mean,
                  Median,
                  Range,
                  Hellinger };

private:
    BR_PROPERTY(Method, method, Mean)
    BR_PROPERTY(QString, inputVariable, "Label")

    Mat a, b; // dst = (src - b) / a

    static void _train(Method method, const cv::Mat &m, const QList<int> &labels, double *ca, double *cb)
    {
        double A = 1, B = 0;
        if      (method == Mean)      mean(m, &A, &B);
        else if (method == Median)    median(m, &A, &B);
        else if (method == Range)     range(m, &A, &B);
        else if (method == Hellinger) hellinger(m, labels, &A, &B);
        else                          qFatal("Invalid method.");
        *ca = A;
        *cb = B;
    }

    void train(const TemplateList &data)
    {
        Mat m;
        OpenCVUtils::toMat(data.data()).convertTo(m, CV_64F);

        const QList<int> labels = data.indexProperty(inputVariable);
        const int dims = m.cols;

        std::vector<Mat> mv, av, bv;
        split(m, mv);
        for (size_t c = 0; c < mv.size(); c++) {
            av.push_back(Mat(1, dims, CV_64FC1));
            bv.push_back(Mat(1, dims, CV_64FC1));
        }

        QFutureSynchronizer<void> futures;
        for (size_t c = 0; c < mv.size(); c++) {
            for (int i=0; i<dims; i++)
                futures.addFuture(QtConcurrent::run(_train, method, mv[c].col(i), labels, &av[c].at<double>(0, i), &bv[c].at<double>(0, i)));
            av[c] = av[c].reshape(1, data.first().m().rows);
            bv[c] = bv[c].reshape(1, data.first().m().rows);
        }
        futures.waitForFinished();

        merge(av, a);
        merge(bv, b);
        a.convertTo(a, data.first().m().type());
        b.convertTo(b, data.first().m().type());
        OpenCVUtils::saveImage(a, Globals->property("CENTER_TRAIN_A").toString());
        OpenCVUtils::saveImage(b, Globals->property("CENTER_TRAIN_B").toString());
    }

    void project(const Template &src, Template &dst) const
    {
        subtract(src, b, dst);
        divide(dst, a, dst);
    }

    void store(QDataStream &stream) const
    {
        stream << a << b;
    }

    void load(QDataStream &stream)
    {
        stream >> a >> b;
    }

    static void mean(const Mat &src, double *a, double *b)
    {
        Scalar mean, stddev;
        meanStdDev(src, mean, stddev);
        *a = stddev[0];
        *b = mean[0];
    }

    static void median(const Mat &src, double *a, double *b)
    {
        QVector<double> vals; vals.reserve(src.rows);
        for (int i=0; i<src.rows; i++)
            vals.append(src.at<double>(i, 0));
        double q1, q3;
        *b = Common::Median(vals, &q1, &q3);
        *a = q3 - q1;
    }

    static void range(const Mat &src, double *a, double *b)
    {
        double min, max;
        minMaxLoc(src, &min, &max);
        *a = max - min;
        *b = min;
    }

    static void hellinger(const Mat &src, const QList<int> &labels, double *a, double *b)
    {
        const QList<float> vals = OpenCVUtils::matrixToVector<float>(src);
        if (vals.size() != labels.size())
            qFatal("Logic error.");

        QVector<float> genuineScores; genuineScores.reserve(vals.size());
        QVector<float> impostorScores; impostorScores.reserve(vals.size()*vals.size()/2);
        for (int i=0; i<vals.size(); i++)
            for (int j=i+1; j<vals.size(); j++)
                if (labels[i] == labels[j]) genuineScores.append(vals[i]-vals[j]);
                else                        impostorScores.append(vals[i]-vals[j]);

        float min, max;
        Common::MinMax(vals, &min, &max);

        double gm, gs, im, is;
        Common::MeanStdDev(genuineScores, &gm, &gs);
        Common::MeanStdDev(impostorScores, &im, &is);

        *a = (max-min)/sqrt(1-sqrt(2*gs*is/(gs*gs+is*is))*exp(-0.25*pow(gm-im,2.0)/(gs*gs+is*is)));
        *b = min;
    }
};

BR_REGISTER(Transform, CenterTransform)

} // namespace br

#include "imgproc/center.moc"
