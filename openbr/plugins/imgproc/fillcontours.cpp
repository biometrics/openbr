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
#include <openbr/core/opencvutils.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Fills contours with white pixels.
 * \author Scott Klum \cite sklum
 */
class FillContoursTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(float epsilon READ get_epsilon WRITE set_epsilon RESET reset_epsilon STORED false)
    Q_PROPERTY(size_t minSize READ get_minSize WRITE set_minSize RESET reset_minSize STORED false)
    BR_PROPERTY(float, epsilon, 0)
    BR_PROPERTY(size_t, minSize, 40)

    void project(const Template &src, Template &dst) const
    {
        dst.m() = Mat::zeros(src.m().rows,src.m().cols,src.m().type());

        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;

        /// Find contours
        findContours(src.m(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

        if (epsilon > 0)
            for(size_t i=0; i<contours.size(); i++)
                approxPolyDP(Mat(contours[i]), contours[i], epsilon, true);

        for(size_t i=0; i<contours.size(); i++)
            if (contours[i].size() > minSize)
                drawContours(dst, contours, i, Scalar(255,255,255), CV_FILLED);
    }
};

BR_REGISTER(Transform, FillContoursTransform)

} // namespace br

#include "imgproc/fillcontours.moc"

