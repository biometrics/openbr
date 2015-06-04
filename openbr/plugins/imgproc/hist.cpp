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
            Mat hist, chan = mv[i];
            // calcHist requires F or U, might as well convert just in case
            if (mv[i].depth() != CV_8U && mv[i].depth() != CV_32F)
                mv[i].convertTo(chan, CV_32F);
            calcHist(&chan, 1, channels, Mat(), hist, 1, histSize, ranges);
            memcpy(m.ptr(i), hist.ptr(), dims * sizeof(float));
        }

        dst += m;
    }
};

BR_REGISTER(Transform, HistTransform)

} // namespace br

#include "imgproc/hist.moc"
