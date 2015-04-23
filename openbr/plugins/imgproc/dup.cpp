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

namespace br
{

/*!
 * \ingroup transforms
 * \brief Duplicates the Template data.
 * \author Josh Klontz \cite jklontz
 */
class DupTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(int n READ get_n WRITE set_n RESET reset_n STORED false)
    Q_PROPERTY(bool dupLandmarks READ get_dupLandmarks WRITE set_dupLandmarks RESET reset_dupLandmarks STORED false)
    BR_PROPERTY(int, n, 1)
    BR_PROPERTY(bool, dupLandmarks, false)

    void project(const Template &src, Template &dst) const
    {
        for (int i=0; i<n; i++)
            dst.merge(src);

        if (dupLandmarks) {
            QList<QPointF> points = src.file.points();
            QList<QRectF> rects = src.file.rects();

            for (int i=1; i<n; i++) {
                dst.file.appendPoints(points);
                dst.file.appendRects(rects);
            }
        }
    }
};

BR_REGISTER(Transform, DupTransform)

} // namespace br

#include "imgproc/dup.moc"
