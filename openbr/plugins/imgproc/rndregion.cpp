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
 * \brief Selects a random region.
 * \author Josh Klontz \cite jklontz
 */
class RndRegionTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        RNG &rng = theRNG();
        float size = rng.uniform(0.2f, 1.f);
        float x = rng.uniform(0.f, 1.f-size);
        float y = rng.uniform(0.f, 1.f-size);

        dst = src.m()(Rect(src.m().cols * x,
                           src.m().rows * y,
                           src.m().cols * size,
                           src.m().rows * size));
    }
};

BR_REGISTER(Transform, RndRegionTransform)

} // namespace br

#include "imgproc/rndregion.moc"
