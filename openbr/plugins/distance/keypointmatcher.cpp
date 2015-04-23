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
 * \brief Wraps OpenCV Key Point Matcher
 * \br_link http://docs.opencv.org/modules/features2d/doc/common_interfaces_of_feature_detectors.html
 * \author Josh Klontz \cite jklontz
 */
class KeyPointMatcherDistance : public UntrainableDistance
{
    Q_OBJECT
    Q_PROPERTY(QString matcher READ get_matcher WRITE set_matcher RESET reset_matcher STORED false)
    Q_PROPERTY(float maxRatio READ get_maxRatio WRITE set_maxRatio RESET reset_maxRatio STORED false)
    BR_PROPERTY(QString, matcher, "BruteForce")
    BR_PROPERTY(float, maxRatio, 0.8)

    Ptr<DescriptorMatcher> descriptorMatcher;

    void init()
    {
        descriptorMatcher = DescriptorMatcher::create(matcher.toStdString());
        if (descriptorMatcher.empty())
            qFatal("Failed to create DescriptorMatcher: %s", qPrintable(matcher));
    }

    float compare(const Mat &a, const Mat &b) const
    {
        if ((a.rows < 2) || (b.rows < 2)) return 0;

        std::vector< std::vector<DMatch> > matches;
        if (a.rows < b.rows) descriptorMatcher->knnMatch(a, b, matches, 2);
        else                 descriptorMatcher->knnMatch(b, a, matches, 2);

        QList<float> distances;
        foreach (const std::vector<DMatch> &match, matches) {
            if (match[0].distance / match[1].distance > maxRatio) continue;
            distances.append(match[0].distance);
        }
        qSort(distances);

        float similarity = 0;
        for (int i=0; i<distances.size(); i++)
            similarity += 1.f/(1+distances[i])/(i+1);
        return similarity;
    }
};

BR_REGISTER(Distance, KeyPointMatcherDistance)

} // namespace br

#include "distance/keypointmatcher.moc"
