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
 * \brief dst = a*src+b
 * \author Josh Klontz \cite jklontz
 */
class MAddTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(double a READ get_a WRITE set_a RESET reset_a STORED false)
    Q_PROPERTY(double b READ get_b WRITE set_b RESET reset_b STORED false)
    BR_PROPERTY(double, a, 1)
    BR_PROPERTY(double, b, 0)

    void project(const Template &src, Template &dst) const
    {
        src.m().convertTo(dst.m(), src.m().depth(), a, b);
    }
};

BR_REGISTER(Transform, MAddTransform)

} // namespace br

#include "imgproc/madd.moc"
