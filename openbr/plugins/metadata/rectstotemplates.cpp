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
 * \brief For each rectangle bounding box in src, a new Template is created.
 * \author Brendan Klare \cite bklare
 */
class RectsToTemplatesTransform : public UntrainableMetaTransform
{
    Q_OBJECT

private:
    void project(const Template &src, Template &dst) const
    {
        Template tOut(src.file);
        QList<float> confidences = src.file.getList<float>("Confidences");
        QList<QRectF> rects = src.file.rects();
        for (int i = 0; i < rects.size(); i++) {
            cv::Mat m(src, OpenCVUtils::toRect(rects[i]));
            Template t(src.file, m);
            t.file.set("Confidence", confidences[i]);
            t.file.clearRects();
            tOut << t;
        }
        dst = tOut;
    }
};

BR_REGISTER(Transform, RectsToTemplatesTransform)

} // namespace br

#include "metadata/rectstotemplates.moc"
