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
 * \brief Set the template's label to the area of the largest convex hull.
 * \author Josh Klontz \cite jklontz
 */
class ConvexHullTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_ENUMS(Approximation)
    Q_PROPERTY(Approximation approximation READ get_approximation WRITE set_approximation RESET reset_approximation STORED false)
    Q_PROPERTY(int minSize READ get_minSize WRITE set_minSize RESET reset_minSize STORED false)

public:
    enum Approximation { None = CV_CHAIN_APPROX_NONE,
                  Simple = CV_CHAIN_APPROX_SIMPLE,
                  L1 = CV_CHAIN_APPROX_TC89_L1,
                  KCOS = CV_CHAIN_APPROX_TC89_KCOS };

private:
    BR_PROPERTY(Approximation, approximation, None)
    BR_PROPERTY(int, minSize, 40)

    void project(const Template &src, Template &dst) const
    {
        dst.m() = Mat::zeros(src.m().rows,src.m().cols,src.m().type());

        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;

        findContours(src.m(), contours, hierarchy, CV_RETR_EXTERNAL, approximation);

        vector<vector<Point> >hull( contours.size() );
        for( size_t i = 0; i < contours.size(); i++ )
            convexHull( Mat(contours[i]), hull[i], false );

        for(size_t i=0; i<contours.size(); i++)
            if (contours[i].size() > (size_t)minSize)
                drawContours(dst, hull, i, Scalar(255,255,255), CV_FILLED);
    }
};

BR_REGISTER(Transform, ConvexHullTransform)

} // namespace br

#include "imgproc/convexhull.moc"

