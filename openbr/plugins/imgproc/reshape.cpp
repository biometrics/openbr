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
 * \brief Reshape each matrix to the specified number of rows.
 * \author Josh Klontz \cite jklontz
 */
class ReshapeTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int rows READ get_rows WRITE set_rows RESET reset_rows STORED false)
    Q_PROPERTY(int channels READ get_channels WRITE set_channels RESET reset_channels STORED false)
    BR_PROPERTY(int, rows, 1)
    BR_PROPERTY(int, channels, -1)

    void project(const Template &src, Template &dst) const
    {
        dst = src.m().reshape(channels == -1 ? src.m().channels() : channels, rows);
    }
};

BR_REGISTER(Transform, ReshapeTransform)

} // namespace br;

#include "imgproc/reshape.moc"
