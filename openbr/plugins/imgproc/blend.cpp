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
 * \brief Alpha-blend matrices
 * \author Josh Klontz \cite jklontz
 */
class BlendTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QList<float> alpha READ get_alpha WRITE set_alpha RESET reset_alpha STORED false)
    BR_PROPERTY(QList<float>, alpha, QList<float>() << 0.5)

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;
        dst.m() = Mat::zeros(src.m().rows, src.m().cols, src.m().type());

        QList<float> a = alpha;

        if (src.size() == 2 && a.size() == 1)
            a << 1-a[0];

        for (int i=0; i<src.size(); i++) {
            const float weight = (a.size() == src.size()) ? a[i] : 1./src.size();
            dst.m() += src[i]*weight;
        }
    }
};

BR_REGISTER(Transform, BlendTransform)

} // namespace br

#include "imgproc/blend.moc"
