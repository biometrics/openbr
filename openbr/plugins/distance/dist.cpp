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
#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup distances
 * \brief Standard Distance metrics
 * \author Josh Klontz \cite jklontz
 */
class DistDistance : public UntrainableDistance
{
    Q_OBJECT
    Q_ENUMS(Metric)
    Q_PROPERTY(Metric metric READ get_metric WRITE set_metric RESET reset_metric STORED false)
    Q_PROPERTY(bool negLogPlusOne READ get_negLogPlusOne WRITE set_negLogPlusOne RESET reset_negLogPlusOne STORED false)

public:
    /*!< */
    enum Metric { Correlation,
                  ChiSquared,
                  Intersection,
                  Bhattacharyya,
                  INF,
                  L1,
                  L2,
                  Cosine,
                  Dot};

private:
    BR_PROPERTY(Metric, metric, L2)
    BR_PROPERTY(bool, negLogPlusOne, true)

    float compare(const Mat &a, const Mat &b) const
    {
        if ((a.size != b.size) ||
            (a.type() != b.type()))
                return -std::numeric_limits<float>::max();

// TODO: this max value is never returned based on the switch / default
        float result = std::numeric_limits<float>::max();
        switch (metric) {
          case Correlation:
            return compareHist(a, b, CV_COMP_CORREL);
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
          case Cosine:
            return cosine(a, b);
          case Dot:
            return a.dot(b);
          default:
            qFatal("Invalid metric");
        }

        if (result != result)
            qFatal("NaN result.");

        return negLogPlusOne ? -log(result+1) : result;
    }

    static float cosine(const Mat &a, const Mat &b)
    {
        float dot = 0;
        float magA = 0;
        float magB = 0;

        for (int row=0; row<a.rows; row++) {
            for (int col=0; col<a.cols; col++) {
                const float target = a.at<float>(row,col);
                const float query = b.at<float>(row,col);
                dot += target * query;
                magA += target * target;
                magB += query * query;
            }
        }

        return dot / (sqrt(magA)*sqrt(magB));
    }
};

BR_REGISTER(Distance, DistDistance)

} // namespace br

#include "distance/dist.moc"
