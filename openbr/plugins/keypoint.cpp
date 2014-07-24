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
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "openbr_internal.h"
#include "openbr/core/opencvutils.h"

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV Key Point Detector
 * \author Josh Klontz \cite jklontz
 */
class KeyPointDetectorTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(QString detector READ get_detector WRITE set_detector RESET reset_detector STORED false)
    BR_PROPERTY(QString, detector, "SIFT")

    Ptr<FeatureDetector> featureDetector;

    void init()
    {
        featureDetector = FeatureDetector::create(detector.toStdString());
        if (featureDetector.empty())
            qFatal("Failed to create KeyPointDetector: %s", qPrintable(detector));
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        std::vector<KeyPoint> keyPoints;
        try {
            featureDetector->detect(src, keyPoints);
        } catch (...) {
            qWarning("Key point detection failed for file %s", qPrintable(src.file.name));
            dst.file.fte = true;
        }

        QList<Rect> rects;
        foreach (const KeyPoint &keyPoint, keyPoints)
            rects.append(Rect(keyPoint.pt.x-keyPoint.size/2, keyPoint.pt.y-keyPoint.size/2, keyPoint.size, keyPoint.size));
        dst.file.setRects(OpenCVUtils::fromRects(rects));
    }
};

BR_REGISTER(Transform, KeyPointDetectorTransform)

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV Key Point Descriptor
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

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV Key Point Matcher
 * \author Josh Klontz \cite jklontz
 */
class KeyPointMatcherDistance : public Distance
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

/*!
 * \ingroup transforms
 * \brief Specialize wrapper OpenCV SIFT wrapper
 * \author Josh Klontz \cite jklontz
 */
class SIFTDescriptorTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int size READ get_size WRITE set_size RESET reset_size STORED false)
    BR_PROPERTY(int, size, 1)

    SIFT sift;

    void project(const Template &src, Template &dst) const
    {
        std::vector<KeyPoint> keyPoints;
        foreach (const QPointF &val, src.file.points())
            keyPoints.push_back(KeyPoint(val.x(), val.y(), size));

        Mat m;
        sift(src, Mat(), keyPoints, m, true);
        m.setTo(0, m<0); // SIFT returns large negative values when it goes off the edge of the image.
        dst += m;
    }
};

BR_REGISTER(Transform, SIFTDescriptorTransform)

/*!
 * \ingroup transforms
 * \brief OpenCV HOGDescriptor wrapper
 * \author Austin Blanton \cite imaus10
 */
class HoGDescriptorTransform : public UntrainableTransform
{
    Q_OBJECT

    HOGDescriptor hog;

    void project(const Template &src, Template &dst) const
    {
        std::vector<float> descriptorVals;
        std::vector<Point> locations;
        Size winStride = Size(0,0);
        Size padding = Size(0,0);
        foreach (const Mat &rect, src) {
            hog.compute(rect, descriptorVals, winStride, padding, locations);
            Mat HoGFeats(descriptorVals, true);
            dst += HoGFeats;
        }
    }
};

BR_REGISTER(Transform, HoGDescriptorTransform)

/*!
 * \ingroup transforms
 * \brief Add landmarks to the template in a grid layout
 * \author Josh Klontz \cite jklontz
 */
class GridTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int rows READ get_rows WRITE set_rows RESET reset_rows STORED false)
    Q_PROPERTY(int columns READ get_columns WRITE set_columns RESET reset_columns STORED false)
    BR_PROPERTY(int, rows, 1)
    BR_PROPERTY(int, columns, 1)

    void project(const Template &src, Template &dst) const
    {
        QList<QPointF> landmarks;
        const float row_step = 1.f * src.m().rows / rows;
        const float column_step = 1.f * src.m().cols / columns;
        for (float y=row_step/2; y<src.m().rows; y+=row_step)
            for (float x=column_step/2; x<src.m().cols; x+=column_step)
                landmarks.append(QPointF(x,y));
        dst = src;
        dst.file.setPoints(landmarks);
    }
};

BR_REGISTER(Transform, GridTransform)

} // namespace br

#include "keypoint.moc"
