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

#include <openbr_plugin.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Subtract two matrices.
 * \author Josh Klontz \cite jklontz
 */
class SubtractTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        if (src.size() != 2) qFatal("Expected exactly two source images, got %d.", src.size());
        dst.file = src.file;
        subtract(src[0], src[1], dst);
    }
};

BR_REGISTER(Transform, SubtractTransform)

/*!
 * \ingroup transforms
 * \brief Take the absolute difference of two matrices.
 * \author Josh Klontz \cite jklontz
 */
class AbsDiffTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        if (src.size() != 2) qFatal("Expected exactly two source images, got %d.", src.size());
        dst.file = src.file;
        absdiff(src[0], src[1], dst);
    }
};

BR_REGISTER(Transform, AbsDiffTransform)

/*!
 * \ingroup transforms
 * \brief Logical AND of two matrices.
 * \author Josh Klontz \cite jklontz
 */
class AndTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;
        dst.m() = src.first();
        for (int i=1; i<src.size(); i++)
            bitwise_and(src[i], dst, dst);
    }
};

BR_REGISTER(Transform, AndTransform)

} // namespace br

#include "reduce.moc"
