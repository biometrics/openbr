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

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Draw a line representing the direction and magnitude of optical flow at the specified points.
 * \author Austin Blanton \cite imaus10
 */
class DrawOpticalFlow : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(QString original READ get_original WRITE set_original RESET reset_original STORED false)
    BR_PROPERTY(QString, original, "original")

    void project(const Template &src, Template &dst) const
    {
        const Scalar color(0,255,0);
        Mat flow = src.m();
        dst = src;
        if (!dst.file.contains(original)) qFatal("The original img must be saved in the metadata with SaveMat.");
        dst.m() = dst.file.get<Mat>(original);
        dst.file.remove(original);
        foreach (const Point2f &pt, OpenCVUtils::toPoints(dst.file.points())) {
            Point2f dxy = flow.at<Point2f>(pt.y, pt.x);
            Point2f newPt(pt.x+dxy.x, pt.y+dxy.y);
            line(dst, pt, newPt, color);
        }
    }
};

BR_REGISTER(Transform, DrawOpticalFlow)

} // namespace br

#include "gui/drawopticalflow.moc"
