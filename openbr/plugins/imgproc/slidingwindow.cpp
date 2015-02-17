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

// Find avg aspect ratio
static float getAspectRatio(const TemplateList &data)
{
    double tempRatio = 0;
    int ratioCnt = 0;

    foreach (const Template &tmpl, data) {
        QList<Rect> posRects = OpenCVUtils::toRects(tmpl.file.rects());
        foreach (const Rect &posRect, posRects) {
            if (posRect.x + posRect.width >= tmpl.m().cols || posRect.y + posRect.height >= tmpl.m().rows || posRect.x < 0 || posRect.y < 0) {
                continue;
            }
            tempRatio += (float)posRect.width / (float)posRect.height;
            ratioCnt += 1;
        }
    }
    return tempRatio / (double)ratioCnt;
}

/*!
 * \ingroup transforms
 * \brief Applies a transform to a sliding window.
 *        Discards negative detections.
 * \author Austin Blanton \cite imaus10
 */
class SlidingWindowTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(br::Transform *transform READ get_transform WRITE set_transform RESET reset_transform STORED false)
    Q_PROPERTY(int windowWidth READ get_windowWidth WRITE set_windowWidth RESET reset_windowWidth STORED false)
    Q_PROPERTY(bool takeFirst READ get_takeFirst WRITE set_takeFirst RESET reset_takeFirst STORED false)
    Q_PROPERTY(float threshold READ get_threshold WRITE set_threshold RESET reset_threshold STORED false)
    Q_PROPERTY(float stepFraction READ get_stepFraction WRITE set_stepFraction RESET reset_stepFraction STORED false)
    Q_PROPERTY(int ignoreBorder READ get_ignoreBorder WRITE set_ignoreBorder RESET reset_ignoreBorder STORED true)
    BR_PROPERTY(br::Transform *, transform, NULL)
    BR_PROPERTY(int, windowWidth, 24)
    BR_PROPERTY(bool, takeFirst, false)
    BR_PROPERTY(float, threshold, 0)
    BR_PROPERTY(float, stepFraction, 0.25)
    BR_PROPERTY(int, ignoreBorder, 0)

private:
    int windowHeight;
    bool skipProject;

    void train(const TemplateList &data)
    {
        skipProject = true;
        float aspectRatio = data.first().file.get<float>("aspectRatio", -1);
        if (aspectRatio == -1)
            aspectRatio = getAspectRatio(data);
        windowHeight = qRound(windowWidth / aspectRatio);

        if (transform->trainable) {
            TemplateList dataOut = data;
            if (ignoreBorder > 0) {
                for (int i = 0; i < dataOut.size(); i++) {
                    Template t = dataOut[i];
                    Mat m = t.m();
                    dataOut.replace(i,Template(t.file, Mat(m,Rect(ignoreBorder,ignoreBorder,m.cols - ignoreBorder * 2, m.rows - ignoreBorder * 2))));
                }
            }
            transform->train(dataOut);
        }
    }

    void store(QDataStream &stream) const
    {
        transform->store(stream);
        stream << windowHeight;
    }

    void load(QDataStream &stream)
    {
        transform->load(stream);
        stream >> windowHeight;
    }

    void project(const Template &src, Template &dst) const
    {
        float scale = src.file.get<float>("scale", 1);
        projectHelp(src, dst, windowWidth, windowHeight, scale);
    }

 protected:
     void projectHelp(const Template &src, Template &dst, int windowWidth, int windowHeight, float scale = 1) const
     {

        dst = src;
        if (skipProject) {
            dst = src;
            return;
        }

        Template windowTemplate(src.file, src);
        QList<float> confidences = dst.file.getList<float>("Confidences", QList<float>());
        for (float y = 0; y + windowHeight < src.m().rows; y += windowHeight*stepFraction) {
            for (float x = 0; x + windowWidth < src.m().cols; x += windowWidth*stepFraction) {
                Mat windowMat(src, Rect(x + ignoreBorder, y + ignoreBorder, windowWidth - ignoreBorder * 2, windowHeight - ignoreBorder * 2));
                windowTemplate.replace(0,windowMat);
                Template detect;
                transform->project(windowTemplate, detect);
                float conf = detect.m().at<float>(0);

                // the result will be in the Label
                if (conf > threshold) {
                    dst.file.appendRect(QRectF(x*scale, y*scale, windowWidth*scale, windowHeight*scale));
                    confidences.append(conf);
                    if (takeFirst)
                        return;
                }
            }
        }
        dst.file.setList<float>("Confidences", confidences);
    }
};

BR_REGISTER(Transform, SlidingWindowTransform)

/*!
 * \ingroup transforms
 * \brief Overloads SlidingWindowTransform for integral images that should be
 *        sampled at multiple scales.
 * \author Josh Klontz \cite jklontz
 */
class IntegralSlidingWindowTransform : public SlidingWindowTransform
{
    Q_OBJECT

 private:
    void project(const Template &src, Template &dst) const
    {
        // TODO: call SlidingWindowTransform::project on multiple scales
        SlidingWindowTransform::projectHelp(src, dst, 24, 24);
    }
};

BR_REGISTER(Transform, IntegralSlidingWindowTransform)

} // namespace br

#include "imgproc/slidingwindow.moc"
