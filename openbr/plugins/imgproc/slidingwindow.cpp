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

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Applies a classifier to a sliding window.
 *        Discards negative detections.
 * \author Jordan Cheney \cite JordanCheney
 */
class SlidingWindowTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(br::Classifier *classifier READ get_classifier WRITE set_classifier RESET reset_classifier STORED false)
    Q_PROPERTY(int minSize READ get_minSize WRITE set_minSize RESET reset_minSize STORED false)
    Q_PROPERTY(int maxSize READ get_maxSize WRITE set_maxSize RESET reset_maxSize STORED false)
    Q_PROPERTY(float scaleFactor READ get_scaleFactor WRITE set_scaleFactor RESET reset_scaleFactor STORED false)

    BR_PROPERTY(br::Classifier *, classifier, NULL)
    BR_PROPERTY(int, minSize, 20)
    BR_PROPERTY(int, maxSize, -1)
    BR_PROPERTY(float, scaleFactor, 1.2)

    void train(const TemplateList &_data)
    {
        QList<Mat> data = _data.data();
        QList<float> labels = File::get<float>(_data, "Label");

        classifier->train(data, labels);
    }

    void project(const Template &src, Template &dst) const
    {
        (void) src;
        (void) dst;
    }
};

BR_REGISTER(Transform, SlidingWindowTransform)


} // namespace br

#include "imgproc/slidingwindow.moc"
