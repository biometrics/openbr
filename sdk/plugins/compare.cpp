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
#include <openbr_plugin.h>

#include "core/distance_sse.h"

using namespace cv;
using namespace br;

/*!
 * \ingroup distances
 * \brief Standard distance metrics
 * \author Josh Klontz \cite jklontz
 */
class Dist : public Distance
{
    Q_OBJECT
    Q_ENUMS(Metric)
    Q_PROPERTY(Metric metric READ get_metric WRITE set_metric RESET reset_metric STORED false)

public:
    /*!< */
    enum Metric { Correlation,
                  ChiSquared,
                  Intersection,
                  Bhattacharyya,
                  INF,
                  L1,
                  L2,
                  CosineSimilarity };

private:
    BR_PROPERTY(Metric, metric, L2)

    float compare(const Template &a, const Template &b) const
    {
        if ((a.m().size != b.m().size) ||
            (a.m().type() != b.m().type()))
                return -std::numeric_limits<float>::max();

        float result = std::numeric_limits<float>::max();
        switch (metric) {
          case Correlation:
            result = -compareHist(a, b, CV_COMP_CORREL);
            break;
          case ChiSquared:
            result = compareHist(a, b, CV_COMP_CHISQR);
            break;
          case Intersection:
            result = compareHist(a, b, CV_COMP_INTERSECT);
            break;
          case Bhattacharyya:
            result = compareHist(a, b, CV_COMP_BHATTACHARYYA);
            break;
          case INF:
            result = norm(a, b, NORM_INF);
            break;
          case L1:
            result = norm(a, b, NORM_L1);
            break;
          case L2:
            result = norm(a, b, NORM_L2);
            break;
          case CosineSimilarity:
            result = cosineSimilarity(a, b);
            break;
          default:
            qFatal("Invalid metric");
        }

        if (result != result)
            qFatal("Dist::compare NaN result.");

        return -log(result+1);
    }

    static float cosineSimilarity(const Mat &a, const Mat &b)
    {
        assert((a.type() == CV_32FC1) && (b.type() == CV_32FC1));
        assert((a.rows == b.rows) && (a.cols == b.cols));

        float denom = 0;
        float tnum = 0;
        float qnum = 0;

        for (int row=0; row<a.rows; row++) {
            for (int col=0; col<a.cols; col++) {
                float target = a.at<float>(row,col);
                float query = b.at<float>(row,col);

                denom += target * query;
                tnum += target * target;
                qnum += query * query;
            }
        }

        return denom / (sqrt(tnum)*sqrt(qnum));
    }
};

BR_REGISTER(Distance, Dist)


/*!
 * \ingroup distances
 * \brief Fast 8-bit L1 distance
 * \author Josh Klontz \cite jklontz
 */
class UCharL1 : public Distance
{
    Q_OBJECT

    float compare(const Template &a, const Template &b) const
    {
        return l1(a.m().data, b.m().data, a.m().total());
    }
};

BR_REGISTER(Distance, UCharL1)


/*!
 * \ingroup distances
 * \brief Fast 4-bit L1 distance
 * \author Josh Klontz \cite jklontz
 */
class PackedUCharL1 : public Distance
{
    Q_OBJECT

    float compare(const Template &a, const Template &b) const
    {
        return packed_l1(a.m().data, b.m().data, a.m().total());
    }
};

BR_REGISTER(Distance, PackedUCharL1)

/*!
 * \ingroup distances
 * \brief Returns \c true if the templates are identical, \c false otherwise.
 * \author Josh Klontz \cite jklontz
 */
class Identical : public Distance
{
    Q_OBJECT

    float compare(const Template &a, const Template &b) const
    {
        const Mat &am = a.m();
        const Mat &bm = b.m();
        const size_t size = am.total() * am.elemSize();
        if (size != bm.total() * bm.elemSize()) return 0;
        for (size_t i=0; i<size; i++)
            if (am.data[i] != bm.data[i]) return 0;
        return 1;
    }
};

BR_REGISTER(Distance, Identical)

#include "compare.moc"
