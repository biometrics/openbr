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

#include <opencv2/features2d/features2d.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV Key Point Descriptor
 * \br_link http://docs.opencv.org/modules/features2d/doc/common_interfaces_of_feature_detectors.html
 * \author Josh Klontz \cite jklontz
 */
class KeyPointDescriptorTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(QString descriptor READ get_descriptor WRITE set_descriptor RESET reset_descriptor STORED false)
    Q_PROPERTY(int size READ get_size WRITE set_size RESET reset_size STORED false)
    BR_PROPERTY(QString, descriptor, "SIFT")
    BR_PROPERTY(int, size, -1)

    Ptr<DescriptorExtractor> descriptorExtractor;

    void init()
    {
        descriptorExtractor = DescriptorExtractor::create(descriptor.toStdString());
        if (descriptorExtractor.empty())
            qFatal("Failed to create DescriptorExtractor: %s", qPrintable(descriptor));
    }

    void project(const Template &src, Template &dst) const
    {
        std::vector<KeyPoint> keyPoints;
        if (size == -1)
            foreach (const QRectF &ROI, src.file.rects())
                keyPoints.push_back(KeyPoint(ROI.x()+ROI.width()/2, ROI.y()+ROI.height()/2, (ROI.width() + ROI.height())/2));
        else
            foreach (const QPointF &landmark, src.file.points())
                keyPoints.push_back(KeyPoint(landmark.x(), landmark.y(), size));
        if (keyPoints.empty()) return;
        descriptorExtractor->compute(src, keyPoints, dst);
    }
};

BR_REGISTER(Transform, KeyPointDescriptorTransform)

} // namespace br

#include "imgproc/keypointdescriptor.moc"
