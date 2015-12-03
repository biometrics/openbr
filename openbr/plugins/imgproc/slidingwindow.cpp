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
#include <openbr/core/qtutils.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Sliding Window Framework for object detection. Performs an exhaustive search of an image by sliding a window of a given size around the image and then resizing the image and repeating until terminating conditions are met.
 * \author Jordan Cheney \cite jcheney
 * \author Scott Klum \cite sklum
 * \br_property Classifier* classifier The classifier that determines if a given window is a positive or negative sample. The size of the window is determined using the classifiers *windowSize* method.
 * \br_property int minSize The smallest sized object to detect in pixels
 * \br_property int maxSize The largest sized object to detect in pixels. A negative value will set maxSize == image size
 * \br_property float scaleFactor The factor to scale the image by during each resize.
 * \br_property float minGroupingConfidence A threshold for positive detections. Positive detections returned by the classifier that have confidences below this threshold are considered negative detections.
 * \br_property float eps Parameter for non-maximum supression
 * \br_property int minNeighbors Parameter for non-maximum supression
 * \br_property bool group If false, non-maxima supression will not be performed
 * \br_property int shrinkingFactor Step value for sliding window
 * \br_property bool clone If false, window will not be cloned (i.e. the representation used by the classifier does not need continuous matrix data)
 */
class SlidingWindowTransform : public MetaTransform
{
    Q_OBJECT

    Q_PROPERTY(br::Classifier* classifier READ get_classifier WRITE set_classifier RESET reset_classifier STORED false)

    Q_PROPERTY(int minSize READ get_minSize WRITE set_minSize RESET reset_minSize STORED false)
    Q_PROPERTY(int maxSize READ get_maxSize WRITE set_maxSize RESET reset_maxSize STORED false)
    Q_PROPERTY(float scaleFactor READ get_scaleFactor WRITE set_scaleFactor RESET reset_scaleFactor STORED false)
    Q_PROPERTY(float minGroupingConfidence READ get_minGroupingConfidence WRITE set_minGroupingConfidence RESET reset_minGroupingConfidence STORED false)
    Q_PROPERTY(float eps READ get_eps WRITE set_eps RESET reset_eps STORED false)
    Q_PROPERTY(float minNeighbors READ get_minNeighbors WRITE set_minNeighbors RESET reset_minNeighbors STORED false)
    Q_PROPERTY(bool group READ get_group WRITE set_group RESET reset_group STORED false)
    Q_PROPERTY(int shrinkingFactor READ get_shrinkingFactor WRITE set_shrinkingFactor RESET reset_shrinkingFactor STORED false)
    Q_PROPERTY(bool clone READ get_clone WRITE set_clone RESET reset_clone STORED false)
    Q_PROPERTY(float minConfidence READ get_minConfidence WRITE set_minConfidence RESET reset_minConfidence STORED false)
    Q_PROPERTY(bool ROCMode READ get_ROCMode WRITE set_ROCMode RESET reset_ROCMode STORED false)
    Q_PROPERTY(QString outputVariable READ get_outputVariable WRITE set_outputVariable RESET reset_outputVariable STORED false)
    BR_PROPERTY(br::Classifier*, classifier, NULL)
    BR_PROPERTY(int, minSize, 20)
    BR_PROPERTY(int, maxSize, -1)
    BR_PROPERTY(float, scaleFactor, 1.2)
    BR_PROPERTY(float, minGroupingConfidence, -std::numeric_limits<float>::max())
    BR_PROPERTY(float, eps, 0.2)
    BR_PROPERTY(int, minNeighbors, 3)
    BR_PROPERTY(bool, group, true)
    BR_PROPERTY(int, shrinkingFactor, 1)
    BR_PROPERTY(bool, clone, true)
    BR_PROPERTY(float, minConfidence, 0)
    BR_PROPERTY(bool, ROCMode, false)
    BR_PROPERTY(QString, outputVariable, "FrontalFace")

    void train(const TemplateList &data)
    {
        classifier->train(data);
    }

    void project(const Template &src, Template &dst) const
    {
        TemplateList temp;
        project(TemplateList() << src, temp);
        if (!temp.isEmpty()) dst = temp.first();
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        foreach (const Template &t, src) {
            // As a special case, skip detection if the appropriate metadata already exists
            if (t.file.contains("Face")) {
                Template u = t;
                u.file.setRects(QList<QRectF>() << t.file.get<QRectF>("Face"));
                u.file.set("Confidence", t.file.get<float>("Confidence", 1));
                dst.append(u);
                continue;
            }

            const bool enrollAll = t.file.getBool("enrollAll");

            // Mirror the behavior of ExpandTransform in the special case
            // of an empty template.
            if (t.empty() && !enrollAll) {
                dst.append(t);
                continue;
            }

            // SlidingWindow assumes that all matricies in a template represent
            // different channels of the same image!
            const Size imageSize = t.m().size();
            const int minSize = t.file.get<int>("MinSize", this->minSize);
            QList<Rect> rects;
            QList<float> confidences;

            int dx, dy;
            const Size classifierSize = classifier->windowSize(&dx, &dy);

            for (double factor = 1; ; factor *= scaleFactor) {
                // TODO: This should support non-square sizes
                // Compute the size of the window in which we will detect faces
                const Size detectionSize(cvRound(minSize*factor),cvRound(minSize*factor));

                // Stop if detection size is bigger than the image itself
                if (detectionSize.width > imageSize.width || detectionSize.height > imageSize.height)
                    break;

                const float widthScale = (float)classifierSize.width/detectionSize.width;
                const float heightScale = (float)classifierSize.height/detectionSize.height;

                // Scale the image such that the detection size within the image corresponds to the respresentation size
                const Size scaledImageSize(cvRound(imageSize.width*widthScale), cvRound(imageSize.height*heightScale));

                Template rep(t.file);
                foreach (const Mat &m, t) {
                    Mat scaledImage;
                    resize(m, scaledImage, scaledImageSize, 0, 0, CV_INTER_AREA);
                    rep.append(scaledImage);
                }
                rep = classifier->preprocess(rep);
		
                // Pre-allocate the window to avoid constructing this every iteration
                Template window(t.file);
                for (int i=0; i<rep.size(); i++)
                    window.append(Mat());

                const int step = factor > 2.0 ? shrinkingFactor : shrinkingFactor*2;
                for (int y = 0; y < scaledImageSize.height-classifierSize.height; y += step) {
                    for (int x = 0; x < scaledImageSize.width-classifierSize.width; x += step) {
                        for (int i=0; i<rep.size(); i++) {
                            if (clone)
                                window[i] = rep[i](Rect(Point(x, y), Size(classifierSize.width+dx, classifierSize.height+dy))).clone();
                            else
                                window[i] = rep[i](Rect(Point(x, y), Size(classifierSize.width+dx, classifierSize.height+dy)));
                        }

                        float confidence = 0;
                        int result = classifier->classify(window, false, &confidence);

                        if (result == 1) {
                            rects.append(Rect(cvRound(x/widthScale), cvRound(y/heightScale), detectionSize.width, detectionSize.height));
                            confidences.append(confidence);
                        } else
                            x += step;
                    }
                }
            }

            if (group)
                OpenCVUtils::group(rects, confidences, minGroupingConfidence, minNeighbors, eps);

            if (!enrollAll && rects.empty()) {
                rects.append(Rect(0, 0, imageSize.width, imageSize.height));
                confidences.append(-std::numeric_limits<float>::max());
            }

            const float minConfidence = t.file.get<float>("MinConfidence", this->minConfidence);
            for (int j=0; j<rects.size(); j++) {
                if (ROCMode || confidences[j] >= minConfidence) {
                    Template u = t;
                    u.file.set("Confidence", confidences[j]);
                    const QRectF rect = OpenCVUtils::fromRect(rects[j]);
                    u.file.appendRect(rect);
                    u.file.set(outputVariable, rect);
                    dst.append(u);
                }
            }
        }
    }

    void load(QDataStream &stream)
    {
        classifier->load(stream);
    }

    void store(QDataStream &stream) const
    {
        classifier->store(stream);
    }
};

BR_REGISTER(Transform, SlidingWindowTransform)

} // namespace br

#include "imgproc/slidingwindow.moc"
