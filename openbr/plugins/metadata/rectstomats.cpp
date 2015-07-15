/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2015 Rank One Computing Corporation                             *
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
 * \brief For each rectangle bounding box in src, the mat is cropped and appended
 *          the the template's list of mats.
 * \author Brendan Klare \cite bklare
 */
class RectsToMatsTransform : public UntrainableTransform
{
    Q_OBJECT

private:
    void project(const Template &src, Template &dst) const
    {
        for (int i = 0; i < src.file.rects().size(); i++) 
            dst += cv::Mat(src, OpenCVUtils::toRect(src.file.rects()[i]));
    }
};

BR_REGISTER(Transform, RectsToMatsTransform)

} // namespace br

#include "metadata/rectstomats.moc"
