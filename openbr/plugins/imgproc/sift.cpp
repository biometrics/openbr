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

#include <opencv2/nonfree/nonfree.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Specialize wrapper OpenCV SIFT wrapper
 * \br_link http://docs.opencv.org/modules/nonfree/doc/feature_detection.html
 * \author Josh Klontz \cite jklontz
 */
class SIFTDescriptorTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int size READ get_size WRITE set_size RESET reset_size STORED false)
    Q_PROPERTY(QList<int> sizes READ get_sizes WRITE set_sizes RESET reset_sizes STORED false)
    Q_PROPERTY(int nFeatures READ get_nFeatures WRITE set_nFeatures RESET reset_nFeatures STORED false)
    Q_PROPERTY(int nOctaveLayers READ get_nOctaveLayers WRITE set_nOctaveLayers RESET reset_nOctaveLayers STORED false)
    Q_PROPERTY(double contrastThreshold READ get_contrastThreshold WRITE set_contrastThreshold RESET reset_contrastThreshold STORED false)
    Q_PROPERTY(double edgeThreshold READ get_edgeThreshold WRITE set_edgeThreshold RESET reset_edgeThreshold STORED false)
    Q_PROPERTY(double sigma READ get_sigma WRITE set_sigma RESET reset_sigma STORED false)
    BR_PROPERTY(int, size, 1)
    BR_PROPERTY(QList<int>, sizes, QList<int>())
    BR_PROPERTY(int, nFeatures, 0)
    BR_PROPERTY(int, nOctaveLayers, 3)
    BR_PROPERTY(double, contrastThreshold, 0.04)
    BR_PROPERTY(double, edgeThreshold, 10)
    BR_PROPERTY(double, sigma, 1.6)

    SIFT sift;

    void init()
    {
        if (sizes.empty())
            sizes.append(size);
        sift = SIFT(nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    }

    void project(const Template &src, Template &dst) const
    {
        std::vector<KeyPoint> keyPoints;
        foreach (const QPointF &val, src.file.points())
            foreach (const int r, sizes)
                keyPoints.push_back(KeyPoint(val.x(), val.y(), r));

        Mat m;
        sift(src, Mat(), keyPoints, m, true);
        m.setTo(0, m<0); // SIFT returns large negative values when it goes off the edge of the image.
        dst += m;
    }
};

BR_REGISTER(Transform, SIFTDescriptorTransform)

} // namespace br

#include "imgproc/sift.moc"
