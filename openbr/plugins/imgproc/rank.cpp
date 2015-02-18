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

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/common.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

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

} // namespace br

#include "imgproc/rank.moc"
