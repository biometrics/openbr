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

#include <opencv2/imgproc/imgproc.hpp>
#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Randomly rotates an image in a specified range.
 * \author Scott Klum \cite sklum
 */
class RndRotateTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(QList<int> range READ get_range WRITE set_range RESET reset_range STORED false)
    Q_PROPERTY(int center READ get_center WRITE set_center RESET reset_center STORED false)
    BR_PROPERTY(QList<int>, range, QList<int>() << -15 << 15)
    BR_PROPERTY(int, center, -1)

    void project(const Template &src, Template &dst) const {
        int span = range.first() - range.last();
        int angle = span == 0 ? range.first() : (rand() % span) + range.first();
        OpenCVUtils::rotate(src, dst, angle);
    }
};

BR_REGISTER(Transform, RndRotateTransform)

} // namespace br

#include "imgproc/rndrotate.moc"
