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
#include <openbr/openbr_plugin.h>

#include "openbr/core/common.h"
#include "openbr/core/opencvutils.h"

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Histograms the matrix
 * \author Josh Klontz \cite jklontz
 */
class HistTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(float max READ get_max WRITE set_max RESET reset_max STORED false)
    Q_PROPERTY(float min READ get_min WRITE set_min RESET reset_min STORED false)
    Q_PROPERTY(int dims READ get_dims WRITE set_dims RESET reset_dims STORED false)
    BR_PROPERTY(float, max, 256)
    BR_PROPERTY(float, min, 0)
    BR_PROPERTY(int, dims, -1)

    void project(const Template &src, Template &dst) const
    {
        const int dims = this->dims == -1 ? max - min : this->dims;

        std::vector<Mat> mv;
        split(src, mv);
        Mat m(mv.size(), dims, CV_32FC1);

        for (size_t i=0; i<mv.size(); i++) {
            int channels[] = {0};
            int histSize[] = {dims};
            float range[] = {min, max};
            const float* ranges[] = {range};
            Mat hist;
            calcHist(&mv[i], 1, channels, Mat(), hist, 1, histSize, ranges);
            memcpy(m.ptr(i), hist.ptr(), dims * sizeof(float));
        }

        dst += m;
    }
};

BR_REGISTER(Transform, HistTransform)

/*!
 * \ingroup transforms
 * \brief Quantizes the values into bins.
 * \author Josh Klontz \cite jklontz
 */
class BinTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(float min READ get_min WRITE set_min RESET reset_min STORED false)
    Q_PROPERTY(float max READ get_max WRITE set_max RESET reset_max STORED false)
    Q_PROPERTY(int bins READ get_bins WRITE set_bins RESET reset_bins STORED false)
    Q_PROPERTY(bool split READ get_split WRITE set_split RESET reset_split STORED false)
    BR_PROPERTY(float, min, 0)
    BR_PROPERTY(float, max, 255)
    BR_PROPERTY(int, bins, 8)
    BR_PROPERTY(bool, split, false)

    void project(const Template &src, Template &dst) const
    {
        const double floor = ((src.m().depth() == CV_32F) || (src.m().depth() == CV_64F)) ? -0.5 : 0;
        src.m().convertTo(dst, bins > 256 ? CV_16U : CV_8U, bins/(max-min), floor);
        if (!split) return;

        Mat input = dst;
        QList<Mat> outputs; outputs.reserve(bins);
        for (int i=0; i<bins; i++)
            outputs.append(input == i); // Note: Matrix elements are 0 or 255
        dst.clear(); dst.append(outputs);
    }
};

BR_REGISTER(Transform, BinTransform)

/*!
 * \ingroup transforms
 * \brief Converts each element to its rank-ordered value.
 * \author Josh Klontz \cite jklontz
 */
class RankTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        const Mat &m = src;
        assert(m.channels() == 1);
        dst = Mat(m.rows, m.cols, CV_32FC1);
        typedef QPair<float,int> Tuple;
        QList<Tuple> tuples = Common::Sort(OpenCVUtils::matrixToVector<float>(m));

        float prevValue = 0;
        int prevRank = 0;
        for (int i=0; i<tuples.size(); i++) {
            int rank;
            if (tuples[i].first == prevValue) rank = prevRank;
            else                              rank = i;
            dst.m().at<float>(tuples[i].second / m.cols, tuples[i].second % m.cols) = rank;
            prevValue = tuples[i].first;
            prevRank = rank;
        }
    }
};

BR_REGISTER(Transform, RankTransform)

/*!
 * \ingroup transforms
 * \brief An integral histogram
 * \author Josh Klontz \cite jklontz
 */
class IntegralHistTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int bins READ get_bins WRITE set_bins RESET reset_bins STORED false)
    Q_PROPERTY(int radius READ get_radius WRITE set_radius RESET reset_radius STORED false)
    BR_PROPERTY(int, bins, 256)
    BR_PROPERTY(int, radius, 16)

    void project(const Template &src, Template &dst) const
    {
        const Mat &m = src.m();
        if (m.type() != CV_8UC1) qFatal("IntegralHist requires 8UC1 matrices.");

        Mat integral(m.rows/radius+1, (m.cols/radius+1)*bins, CV_32SC1);
        integral.setTo(0);
        for (int i=1; i<integral.rows; i++) {
            for (int j=1; j<integral.cols; j+=bins) {
                for (int k=0; k<bins; k++) integral.at<qint32>(i, j+k) += integral.at<qint32>(i-1, j     +k);
                for (int k=0; k<bins; k++) integral.at<qint32>(i, j+k) += integral.at<qint32>(i  , j-bins+k);
                for (int k=0; k<bins; k++) integral.at<qint32>(i, j+k) -= integral.at<qint32>(i-1, j-bins+k);
                for (int k=0; k<radius; k++)
                    for (int l=0; l<radius; l++)
                        integral.at<qint32>(i, j+m.at<quint8>((i-1)*radius+k,(j/bins-1)*radius+l))++;
            }
        }
        dst = integral;
    }
};

BR_REGISTER(Transform, IntegralHistTransform)

} // namespace br

#include "hist.moc"
