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
 * \brief Computes the mean of a set of templates.
 *
 * Suitable for visualization only as it sets every projected template to the mean template.
 *
 * \author Scott Klum \cite sklum
 */
class MeanTransform : public Transform
{
    Q_OBJECT

    Mat mean;

    void train(const TemplateList &data)
    {
        mean = Mat::zeros(data[0].m().rows,data[0].m().cols,CV_32F);

        for (int i = 0; i < data.size(); i++) {
            Mat converted;
            data[i].m().convertTo(converted, CV_32F);
            mean += converted;
        }

        mean /= data.size();
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        dst.m() = mean;
    }

};

BR_REGISTER(Transform, MeanTransform)

} // namespace br

#include "imgproc/mean.moc"
