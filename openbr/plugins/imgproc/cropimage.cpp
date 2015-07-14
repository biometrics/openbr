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

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Crops the image using the bounding box. If multiple bounding boxes existing 
 * 		all such crops will be appended.
 * \author Brendan Klare \cite bklare
 */
class CropImageTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
	for (int i = 0; i < src.file.rects().size(); i++) 
        dst += Mat(src, Rect(src.file.rects()[i].x(), src.file.rects()[i].y(), src.file.rects()[i].width(), src.file.rects()[i].height()));
    }
};

BR_REGISTER(Transform, CropImageTransform)

} // namespace br

#include "imgproc/cropimage.moc"
