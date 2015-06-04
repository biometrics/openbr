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
 * \brief Fill black pixels with the specified color.
 * \author Josh Klontz \cite jklontz
 */
class FloodTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int r READ get_r WRITE set_r RESET reset_r STORED false)
    Q_PROPERTY(int g READ get_g WRITE set_g RESET reset_g STORED false)
    Q_PROPERTY(int b READ get_b WRITE set_b RESET reset_b STORED false)
    Q_PROPERTY(bool all READ get_all WRITE set_all RESET reset_all STORED false)
    BR_PROPERTY(int, r, 0)
    BR_PROPERTY(int, g, 0)
    BR_PROPERTY(int, b, 0)
    BR_PROPERTY(bool, all, false)

    void project(const Template &src, Template &dst) const
    {
        dst = src.m().clone();
        dst.m().setTo(Scalar(r, g, b), all ? Mat() : dst.m()==0);
    }
};

BR_REGISTER(Transform, FloodTransform)

} // namespace br

#include "imgproc/flood.moc"
