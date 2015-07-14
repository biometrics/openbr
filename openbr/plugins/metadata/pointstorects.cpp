/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2015 Rank One Computing Corporation                             *
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
 * \brief For each point, add a rectangle with radius as a half width
 * \author Brendan Klare \cite bklare
 */

class PointsToRectsTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(float radius READ get_radius WRITE set_radius RESET reset_radius STORED false)
    Q_PROPERTY(bool clearRects READ get_clearRects WRITE set_clearRects RESET reset_clearRects STORED false)
    BR_PROPERTY(float, radius, 4)
    BR_PROPERTY(bool, clearRects, true)

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        if (clearRects)
            dst.file.clearRects();

        if (src.file.points().isEmpty()) {
            if (Globals->verbose) qWarning("No landmarks");
            return;
        }

        for (int i = 0; i < src.file.points().size(); i++) {
            dst.file.appendRect(QRectF(src.file.points()[i].x() - radius, src.file.points()[i].y() - radius, radius * 2, radius * 2));
        }
    }
};

BR_REGISTER(Transform, PointsToRectsTransform)

} // namespace br

#include "metadata/pointstorects.moc"
