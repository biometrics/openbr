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

#include <QtConcurrentRun>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Core>
#include <openbr_plugin.h>

#include "core/common.h"
#include "core/opencvutils.h"

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Histogram equalization
 * \author Josh Klontz \cite jklontz
 */
class EqualizeHistTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        equalizeHist(src, dst);
    }
};

BR_REGISTER(Transform, EqualizeHistTransform)

/*!
 * \ingroup transforms
 * \brief Normalize matrix to unit length
 * \author Josh Klontz \cite jklontz
 */
class NormalizeTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_ENUMS(NormType)
    Q_PROPERTY(NormType normType READ get_normType WRITE set_normType RESET reset_normType STORED false)

public:
    /*!< */
    enum NormType { Inf = NORM_INF,
                    L1 = NORM_L1,
                    L2 = NORM_L2 };

private:
    BR_PROPERTY(NormType, normType, L2)

    void project(const Template &src, Template &dst) const
    {
        normalize(src, dst, 1, 0, normType, CV_32F);
    }
};

BR_REGISTER(Transform, NormalizeTransform)

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

public:
    /*!< */
    enum Method { Mean,
                  Median,
                  Range };

private:
    BR_PROPERTY(Method, method, Mean)

    Mat a, b; // dst = (src - b) / a

    static void _train(Method method, const cv::Mat &m, Mat *ca, Mat *cb, int i)
    {
        double A = 1, B = 0;
        if      (method == Mean)   mean(m.col(i), &A, &B);
        else if (method == Median) median(m.col(i), &A, &B);
        else if (method == Range)  range(m.col(i), &A, &B);
        else                       qFatal("Invalid method.");
        ca->at<double>(0, i) = A;
        cb->at<double>(0, i) = B;
    }

    void train(const TemplateList &data)
    {
        Mat m;
        OpenCVUtils::toMat(data.data()).convertTo(m, CV_64F);
        const int dims = m.cols;

        vector<Mat> mv, av, bv;
        split(m, mv);
        for (size_t c = 0; c < mv.size(); c++) {
            av.push_back(Mat(1, dims, CV_64FC1));
            bv.push_back(Mat(1, dims, CV_64FC1));
        }

        QList< QFuture<void> > futures; futures.reserve(dims);
        const bool parallel = (data.size() > 1000) && Globals->parallelism;

        for (size_t c = 0; c < mv.size(); c++) {
            for (int i=0; i<dims; i++)
                if (parallel) futures.append(QtConcurrent::run(_train, method, mv[c], &av[c], &bv[c], i));
                else                                           _train (method, mv[c], &av[c], &bv[c], i);
            av[c] = av[c].reshape(1, data.first().m().rows);
            bv[c] = bv[c].reshape(1, data.first().m().rows);
        }

        if (parallel) Globals->trackFutures(futures);

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
};

BR_REGISTER(Transform, CenterTransform)

/*!
 * \ingroup transforms
 * \brief Remove the row-wise training set average.
 * \author Josh Klontz \cite jklontz
 */
class RowWiseMeanCenterTransform : public Transform
{
    Q_OBJECT
    Mat mean;

    void train(const TemplateList &data)
    {
        Mat m = OpenCVUtils::toMatByRow(data.data());
        mean = Mat(1, m.cols, m.type());
        for (int i=0; i<m.cols; i++)
            mean.col(i) = cv::mean(m.col(i));
    }

    void project(const Template &src, Template &dst) const
    {
        Mat m = src.m().clone();
        for (int i=0; i<m.rows; i++)
            m.row(i) -= mean;
        dst = m;
    }

    void store(QDataStream &stream) const
    {
        stream << mean;
    }

    void load(QDataStream &stream)
    {
        stream >> mean;
    }
};

BR_REGISTER(Transform, RowWiseMeanCenterTransform)

} // namespace br

#include "normalize.moc"
