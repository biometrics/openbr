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
 * \brief Wraps OpenCV merge
 * \br_link http://docs.opencv.org/modules/core/doc/operations_on_arrays.html#merge
 * \author Josh Klontz \cite jklontz
 */
class MergeTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;
        std::vector<Mat> mv;
        foreach (const Mat &m, src)
            mv.push_back(m);
        merge(mv, dst);
    }
};

BR_REGISTER(Transform, MergeTransform)

} // namespace br

#include "imgproc/merge.moc"
