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
class LargestConvexAreaTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(QString outputVariable READ get_outputVariable WRITE set_outputVariable RESET reset_outputVariable STORED false)
    BR_PROPERTY(QString, outputVariable, "Label")

    void project(const Template &src, Template &dst) const
    {
        std::vector< std::vector<Point> > contours;
        findContours(src.m().clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        double maxArea = 0;
        foreach (const std::vector<Point> &contour, contours) {
            std::vector<Point> hull;
            convexHull(contour, hull);
            double area = contourArea(contour);
            double hullArea = contourArea(hull);
            if (area / hullArea > 0.98)
                maxArea = std::max(maxArea, area);
        }
        dst.file.set(outputVariable, maxArea);
    }
};

BR_REGISTER(Transform, LargestConvexAreaTransform)

} // namespace br

#include "imgproc/largestconvexarea.moc"
