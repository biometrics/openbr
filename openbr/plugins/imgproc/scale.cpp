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
 * \brief Scales using the given factor
 * \author Scott Klum \cite sklum
 */
class ScaleTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(float scaleFactor READ get_scaleFactor WRITE set_scaleFactor RESET reset_scaleFactor STORED false)
    BR_PROPERTY(float, scaleFactor, 1.)

    void project(const Template &src, Template &dst) const
    {
        resize(src, dst, Size(src.m().cols*scaleFactor,src.m().rows*scaleFactor));

        QList<QRectF> rects = src.file.rects();
        for (int i=0; i<rects.size(); i++)
            rects[i] = QRectF(rects[i].topLeft()*scaleFactor,rects[i].bottomRight()*scaleFactor);
        dst.file.setRects(rects);

        QList<QPointF> points = src.file.points();
        for (int i=0; i<points.size(); i++)
            points[i] = points[i] * scaleFactor;
        dst.file.setPoints(points);

    }
};

BR_REGISTER(Transform, ScaleTransform)

} // namespace br

#include "imgproc/scale.moc"
