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
 * \brief Flips the image about an axis.
 * \author Josh Klontz \cite jklontz
 */
class FlipTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_ENUMS(Axis)
    Q_PROPERTY(Axis axis READ get_axis WRITE set_axis RESET reset_axis STORED false)
    Q_PROPERTY(bool flipPoints READ get_flipPoints WRITE set_flipPoints RESET reset_flipPoints STORED false)
    Q_PROPERTY(bool flipRects READ get_flipRects WRITE set_flipRects RESET reset_flipRects STORED false)

public:
    enum Axis { X = OpenCVUtils::X,
                Y = OpenCVUtils::Y,
                Both = OpenCVUtils::Both };

private:
    BR_PROPERTY(Axis, axis, Y)
    BR_PROPERTY(bool, flipPoints, true)
    BR_PROPERTY(bool, flipRects, true)

    void project(const Template &src, Template &dst) const
    {
        OpenCVUtils::flip(src, dst, static_cast<OpenCVUtils::Axis>(axis), true, flipPoints, flipRects);
    }
};

BR_REGISTER(Transform, FlipTransform)

} // namespace br

#include "imgproc/flip.moc"
