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

static TemplateList cropTrainingSamples(const TemplateList &data, const float aspectRatio, const int minSize = 32, const float maxOverlap = 0.5, const int negToPosRatio = 1)
{
    TemplateList result;
    foreach (const Template &tmpl, data) {
        QList<Rect> posRects = OpenCVUtils::toRects(tmpl.file.rects());
        QList<Rect> negRects;
        for (int i=0; i<posRects.size(); i++) {
            Rect &posRect = posRects[i];

            // Adjust for training samples that have different aspect ratios
            const int diff = int(posRect.height * aspectRatio) - posRect.width;
            posRect.x -= diff / 2;
            posRect.width += diff;

            // Ignore samples larger than the image
            if ((posRect.x + posRect.width >= tmpl.m().cols) ||
                (posRect.y + posRect.height >= tmpl.m().rows) ||
                (posRect.x < 0) ||
                (posRect.y < 0))
                continue;

            result += Template(tmpl.file, Mat(tmpl, posRect));
            result.last().file.set("Label", QString("pos"));

            // Add random negative samples
            Mat m = tmpl.m();
            int sample = 0;
            while (sample < negToPosRatio) {
                const int x = rand() % m.cols;
                const int y = rand() % m.rows;
                const int maxWidth = m.cols - x;
                const int maxHeight = m.rows - y;
                if (maxWidth <= minSize || maxHeight <= minSize)
                    continue;

                int height;
                int width;
                if (aspectRatio > (float) maxWidth / (float) maxHeight) {
                    width = rand() % (maxWidth - minSize) + minSize;
                    height = qRound(width / aspectRatio);
                } else {
                    height = rand() % (maxHeight - minSize) + minSize;
                    width = qRound(height * aspectRatio);
                }
                Rect negRect(x, y, width, height);

                // The negative samples cannot overlap the positive samples at
                // all, but they may partially overlap with other negatives.
                if (OpenCVUtils::overlaps(posRects, negRect, 0) ||
                    OpenCVUtils::overlaps(negRects, negRect, maxOverlap))
                    continue;

                result += Template(tmpl.file, Mat(tmpl, negRect));
                result.last().file.set("Label", QString("neg"));
                sample++;
            }
        }
    }

    return result;
}

/*!
 * \ingroup transforms
 * \brief DOCUMENT ME
 * \author Austin Blanton \cite imaus10
 */
class BuildScalesTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(br::Transform *transform READ get_transform WRITE set_transform RESET reset_transform STORED false)
    Q_PROPERTY(double scaleFactor READ get_scaleFactor WRITE set_scaleFactor RESET reset_scaleFactor STORED false)
    Q_PROPERTY(bool takeLargestScale READ get_takeLargestScale WRITE set_takeLargestScale RESET reset_takeLargestScale STORED false)
    Q_PROPERTY(int windowWidth READ get_windowWidth WRITE set_windowWidth RESET reset_windowWidth STORED false)
    Q_PROPERTY(int negToPosRatio READ get_negToPosRatio WRITE set_negToPosRatio RESET reset_negToPosRatio STORED false)
    Q_PROPERTY(int minSize READ get_minSize WRITE set_minSize RESET reset_minSize STORED false)
    Q_PROPERTY(double maxOverlap READ get_maxOverlap WRITE set_maxOverlap RESET reset_maxOverlap STORED false)
    Q_PROPERTY(float minScale READ get_minScale WRITE set_minScale RESET reset_minScale STORED false)
    BR_PROPERTY(br::Transform *, transform, NULL)
    BR_PROPERTY(double, scaleFactor, 0.75)
    BR_PROPERTY(bool, takeLargestScale, false)
    BR_PROPERTY(int, windowWidth, 24)
    BR_PROPERTY(int, negToPosRatio, 1)
    BR_PROPERTY(int, minSize, 8)
    BR_PROPERTY(double, maxOverlap, 0)
    BR_PROPERTY(float, minScale, 1.0)

private:
    float aspectRatio;
    int windowHeight;
    bool skipProject;

    void train(const TemplateList &data)
    {
        skipProject = true;
        aspectRatio = getAspectRatio(data);
        windowHeight = qRound(windowWidth / aspectRatio);
        if (transform->trainable) {
            TemplateList full;
            foreach (const Template &roi, cropTrainingSamples(data, aspectRatio, minSize, maxOverlap, negToPosRatio)) {
                Mat resized;
                resize(roi, resized, Size(windowWidth, windowHeight));
                full += Template(roi.file, resized);
            }
            full.first().file.set("aspectRatio", aspectRatio);
            transform->train(full);
        }
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        if (skipProject) {
            dst = src;
            return;
        }

        int rows = src.m().rows;
        int cols = src.m().cols;
        int windowHeight = (int) qRound((float) windowWidth / aspectRatio);

        float startScale;
        if ((cols / rows) > aspectRatio)
            startScale = qRound((float) rows / (float) windowHeight);
        else
            startScale = qRound((float) cols / (float) windowWidth);

        for (float scale = startScale; scale >= minScale; scale -= (1.0 - scaleFactor)) {
            Template scaleImg(dst.file, Mat());
            scaleImg.file.set("scale", scale);
            resize(src, scaleImg, Size(qRound(cols / scale), qRound(rows / scale)));
            transform->project(scaleImg, dst);
            if (takeLargestScale && !dst.file.rects().empty())
                return;
        }
    }

    void store(QDataStream &stream) const
    {
        transform->store(stream);
        stream << aspectRatio << windowHeight;
    }
    void load(QDataStream &stream)
    {
        transform->load(stream);
        stream >> aspectRatio >> windowHeight;
    }
};

BR_REGISTER(Transform, BuildScalesTransform)

} // namespace br

#include "imgproc/multiscale.moc"
