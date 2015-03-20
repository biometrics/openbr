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

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Generates a random landmark.
 * \author Josh Klontz \cite jklontz
 */
class RndPointTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(float x READ get_x WRITE set_x RESET reset_x)
    Q_PROPERTY(float y READ get_y WRITE set_y RESET reset_y)
    BR_PROPERTY(float, x, -1)
    BR_PROPERTY(float, y, -1)

    void train(const TemplateList &data)
    {
        (void) data;

        RNG &rng = theRNG();
        x = rng.uniform(0.f, 1.f);
        y = rng.uniform(0.f, 1.f);
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        dst.file.appendPoint(QPointF(src.m().cols * x, src.m().rows * y));
    }
};

BR_REGISTER(Transform, RndPointTransform)

} // namespace br

#include "imgproc/rndpoint.moc"
