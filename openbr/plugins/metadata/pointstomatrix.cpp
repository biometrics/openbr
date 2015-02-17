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

namespace br
{

/*!
 * \ingroup transforms
 * \brief Converts either the file::points() list or a QList<QPointF> metadata item to be the template's matrix
 * \author Scott Klum \cite sklum
 */
class PointsToMatrixTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED false)
    BR_PROPERTY(QString, inputVariable, QString())

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        if (inputVariable.isEmpty()) {
            dst.m() = OpenCVUtils::pointsToMatrix(dst.file.points());
        } else {
            if (src.file.contains(inputVariable))
                dst.m() = OpenCVUtils::pointsToMatrix(dst.file.get<QList<QPointF> >(inputVariable));
        }
    }
};

BR_REGISTER(Transform, PointsToMatrixTransform)

} // namespace br

#include "metadata/pointstomatrix.moc"
