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
 * \brief Creates a Delaunay triangulation based on a set of points
 * \author Scott Klum \cite sklum
 */
class DrawDelaunayTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        if (src.file.contains("DelaunayTriangles")) {
            QList<Point2f> validTriangles = OpenCVUtils::toPoints(src.file.getList<QPointF>("DelaunayTriangles"));

            // Clone the matrix do draw on it
            for (int i = 0; i < validTriangles.size(); i+=3) {
                line(dst, validTriangles[i], validTriangles[i+1], Scalar(0,0,0), 1);
                line(dst, validTriangles[i+1], validTriangles[i+2], Scalar(0,0,0), 1);
                line(dst, validTriangles[i+2], validTriangles[i], Scalar(0,0,0), 1);
            }
        } else qWarning("Template does not contain Delaunay triangulation.");
    }
};

BR_REGISTER(Transform, DrawDelaunayTransform)

} // namespace br

#include "gui/drawdelaunay.moc"
