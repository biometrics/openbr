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

namespace br
{

/*!
 * \ingroup transforms
 * \brief Check if a the automated face detection contains the landmarks correpsonding 
 *          to the tempate metadata. If not, drop the template. This is meant for use
 *          during training, where the landmarks will be ground truth'd. If one wants to 
 *          using a ground truth bounding box instead, then convert the BB to a landmark.
 * \br_property int index Index of the landmark to be used.          
 * \author Brendan Klare \cite bklare
 */
class VerifyDetectionTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(int index READ get_index WRITE set_index RESET reset_index STORED false)
    BR_PROPERTY(int, index, 14)

    void project(const Template &, Template &dst) const
    {
        qWarning("Single template project not supported by VerifyDetectionTransform.");
        dst.file.fte = true;
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        for (int i = 0; i < src.size(); i++) 
            if (src[i].file.get<QRectF>("FrontalFace").contains(src[i].file.points()[index]))
                dst.append(src[i]);
    }
};

BR_REGISTER(Transform, VerifyDetectionTransform)

} // namespace br

#include "metadata/verifydetection.moc"
