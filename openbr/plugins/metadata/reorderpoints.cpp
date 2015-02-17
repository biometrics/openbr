#include <openbr/plugins/openbr_internal.h>
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

#include <openbr/core/common.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Reorder the points such that points[from[i]] becomes points[to[i]] and
 *        vice versa
 * \author Scott Klum \cite sklum
 */
class ReorderPointsTransform : public UntrainableMetadataTransform
{
    Q_OBJECT

    Q_PROPERTY(QList<int> from READ get_from WRITE set_from RESET reset_from STORED false)
    Q_PROPERTY(QList<int> to READ get_to WRITE set_to RESET reset_to STORED false)
    BR_PROPERTY(QList<int>, from, QList<int>())
    BR_PROPERTY(QList<int>, to, QList<int>())

    void projectMetadata(const File &src, File &dst) const
    {
        if (from.size() == to.size()) {
            QList<QPointF> points = src.points();
            int size = src.points().size();
            if (!points.contains(QPointF(-1,-1)) && Common::Max(from) < size && Common::Max(to) < size) {
                for (int i=0; i<from.size(); i++) {
                    std::swap(points[from[i]],points[to[i]]);
                }
                dst.setPoints(points);
            }
        } else qFatal("Inconsistent sizes for to and from index lists.");
    }
};

BR_REGISTER(Transform, ReorderPointsTransform)

} // namespace br

#include "metadata/reorderpoints.moc"
