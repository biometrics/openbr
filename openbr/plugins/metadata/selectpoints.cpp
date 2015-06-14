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
 * \brief Retains only landmarks/points at the provided indices
 * \author Brendan Klare \cite bklare
 */
class SelectPointsTransform : public UntrainableMetadataTransform
{
    Q_OBJECT
    Q_PROPERTY(QList<int> indices READ get_indices WRITE set_indices RESET reset_indices STORED false)
    Q_PROPERTY(bool invert READ get_invert WRITE set_invert RESET reset_invert STORED false) // keep the points _not_ in the list
    Q_PROPERTY(int rangeStart READ get_rangeStart WRITE set_rangeStart RESET reset_rangeStart STORED false)
    Q_PROPERTY(int rangeEnd READ get_rangeEnd WRITE set_rangeEnd RESET reset_rangeEnd STORED false)
    BR_PROPERTY(QList<int>, indices, QList<int>())
    BR_PROPERTY(bool, invert, false)
    BR_PROPERTY(int, rangeStart, -1)
    BR_PROPERTY(int, rangeEnd, -1)

    void init()
    {
        if ((rangeStart != -1) && (rangeEnd != -1))
            for (int i=rangeStart; i<=rangeEnd; i++)
                if (!indices.contains(i))
                    indices.append(i);
    }

    void projectMetadata(const File &src, File &dst) const
    {
        const QList<QPointF> srcPoints = src.points();
        QList<QPointF> dstPoints;
        for (int i=0; i<srcPoints.size(); i++)
            if (indices.contains(i) ^ invert)
                dstPoints.append(srcPoints[i]);
        dst.setPoints(dstPoints);
    }
};

BR_REGISTER(Transform, SelectPointsTransform)

} // namespace br

#include "metadata/selectpoints.moc"
