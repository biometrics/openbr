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
 * \brief Concatenates all input matrices by column into a single matrix.
 * Use after a fork to concatenate two feature matrices by column.
 * \author Austin Blanton \cite imaus10
 */
class CatColsTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        int half = src.size()/2;
        for (int i=0; i<half; i++) {
            Mat first = src[i];
            Mat second = src[half+i];
            Mat both;
            hconcat(first, second, both);
            dst.append(both);
        }
        dst.file = src.file;
    }
};

BR_REGISTER(Transform, CatColsTransform)

} // namespace br

#include "imgproc/catcols.moc"
