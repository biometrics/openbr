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
 * \brief Randomly rotates an image in a specified range.
 * \author Scott Klum \cite sklum
 */
class RndRotateTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(QList<int> range READ get_range WRITE set_range RESET reset_range STORED false)
    BR_PROPERTY(QList<int>, range, QList<int>() << -15 << 15)

    void project(const Template &src, Template &dst) const {
        int span = range.first() - range.last();
        int angle = (rand() % span) + range.first();
        Mat rotMatrix = getRotationMatrix2D(Point2f(src.m().rows/2,src.m().cols/2),angle,1.0);
        warpAffine(src,dst,rotMatrix,Size(src.m().cols,src.m().rows),INTER_LINEAR,BORDER_REFLECT_101);

        QList<QPointF> points = src.file.points();
        QList<QPointF> rotatedPoints;
        for (int i=0; i<points.size(); i++) {
            rotatedPoints.append(QPointF(points.at(i).x()*rotMatrix.at<double>(0,0)+
                                         points.at(i).y()*rotMatrix.at<double>(0,1)+
                                         rotMatrix.at<double>(0,2),
                                         points.at(i).x()*rotMatrix.at<double>(1,0)+
                                         points.at(i).y()*rotMatrix.at<double>(1,1)+
                                         rotMatrix.at<double>(1,2)));
        }

        dst.file.setPoints(rotatedPoints);
    }
};

BR_REGISTER(Transform, RndRotateTransform)

} // namespace br

#include "imgproc/rndrotate.moc"
