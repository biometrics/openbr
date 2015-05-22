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
#include <openbr/core/cascade.h>
#include <openbr/core/opencvutils.h>
#include <openbr/core/qtutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Sliding Window Framework
 * \author Jordan Cheney
 */

class SlidingWindowTransform : public MetaTransform
{
    Q_OBJECT

    Q_PROPERTY(br::Classifier *classifier READ get_classifier WRITE set_classifier RESET reset_classifier STORED false)

    Q_PROPERTY(int minSize READ get_minSize WRITE set_minSize RESET reset_minSize STORED false)
    Q_PROPERTY(int maxSize READ get_maxSize WRITE set_maxSize RESET reset_maxSize STORED false)
    Q_PROPERTY(float scaleFactor READ get_scaleFactor WRITE set_scaleFactor RESET reset_scaleFactor STORED false)
    Q_PROPERTY(int minNeighbors READ get_minNeighbors WRITE set_minNeighbors RESET reset_minNeighbors STORED false)
    Q_PROPERTY(float eps READ get_eps WRITE set_eps RESET reset_eps STORED false)

    Q_PROPERTY(QString model READ get_model WRITE set_model RESET reset_model STORED false)

    BR_PROPERTY(br::Classifier *, classifier, NULL)
    BR_PROPERTY(int, minSize, 20)
    BR_PROPERTY(int, maxSize, -1)
    BR_PROPERTY(float, scaleFactor, 1.2)
    BR_PROPERTY(int, minNeighbors, 5)
    BR_PROPERTY(float, eps, 0.2)

    BR_PROPERTY(QString, model, "")

    void init()
    {
        QDataStream stream;
        load(stream);
    }

    void train(const TemplateList &data)
    {
        classifier->train(data.data(), File::get<float>(data, "Label", -1));
    }

    void project(const Template &src, Template &dst) const
    {
        TemplateList temp;
        project(TemplateList() << src, temp);
        if (!temp.isEmpty()) dst = temp.first();
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        Size minObjectSize(minSize, minSize);
        Size maxObjectSize;

        foreach (const Template &t, src) {
            const bool enrollAll = t.file.getBool("enrollAll");

            // Mirror the behavior of ExpandTransform in the special case
            // of an empty template.
            if (t.empty() && !enrollAll) {
                dst.append(t);
                continue;
            }

            for (int i=0; i<t.size(); i++) {
                Mat m;
                OpenCVUtils::cvtUChar(t[i], m);
                std::vector<Rect> rects;
                std::vector<int> rejectLevels;
                std::vector<double> levelWeights;

                if (maxObjectSize.height == 0 || maxObjectSize.width == 0)
                    maxObjectSize = m.size();

                Mat imageBuffer(m.rows + 1, m.cols + 1, CV_8U);

                for (double factor = 1; ; factor *= scaleFactor) {
                    int dx, dy;
                    Size originalWindowSize = classifier->windowSize(dx, dy);

                    Size windowSize(cvRound(originalWindowSize.width*factor), cvRound(originalWindowSize.height*factor) );
                    Size scaledImageSize(cvRound(m.cols/factor ), cvRound(m.rows/factor));
                    Size processingRectSize(scaledImageSize.width - originalWindowSize.width, scaledImageSize.height - originalWindowSize.height);

                    if (processingRectSize.width <= 0 || processingRectSize.height <= 0)
                        break;
                    if (windowSize.width > maxObjectSize.width || windowSize.height > maxObjectSize.height)
                        break;
                    if (windowSize.width < minObjectSize.width || windowSize.height < minObjectSize.height)
                        continue;

                    Mat scaledImage(scaledImageSize, CV_8U, imageBuffer.data);
                    resize(m, scaledImage, scaledImageSize, 0, 0, CV_INTER_LINEAR);

                    Mat repImage = classifier->preprocess(scaledImage);

                    int step = factor > 2. ? 1 : 2;
                    for (int y = 0; y < processingRectSize.height; y += step) {
                        for (int x = 0; x < processingRectSize.width; x += step) {
                            Mat window = repImage(Rect(Point(x, y), Size(originalWindowSize.width + dx, originalWindowSize.height + dy))).clone();

                            float gypWeight;
                            int result = classifier->classify(window, gypWeight);

                            if (12 - result < 4) {
                                rects.push_back(Rect(cvRound(x*factor), cvRound(y*factor), windowSize.width, windowSize.height));
                                rejectLevels.push_back(result);
                                levelWeights.push_back(gypWeight);
                            }

                            if (result == 0)
                                x += step;
                        }
                    }
                }

                groupRectangles(rects, rejectLevels, levelWeights, minNeighbors, eps);

                if (!enrollAll && rects.empty())
                    rects.push_back(Rect(0, 0, m.cols, m.rows));

                for (size_t j=0; j<rects.size(); j++) {
                    Template u(t.file, m);
                    if (rejectLevels.size() > j)
                        u.file.set("Confidence", rejectLevels[j]*levelWeights[j]);
                    else
                        u.file.set("Confidence", 1);
                    const QRectF rect = OpenCVUtils::fromRect(rects[j]);
                    u.file.appendRect(rect);
                    u.file.set(model, rect);
                    dst.append(u);
                }
            }
        }
    }

    void load(QDataStream &stream)
    {
        (void)stream;

        QString filename = Globals->sdkPath + "/share/openbr/models/openbrcascades/" + model + "/cascade.xml";
        FileStorage fs(filename.toStdString(), FileStorage::READ);
        if (!fs.isOpened())
            return;

        classifier->read(fs.getFirstTopLevelNode());
    }

    void store(QDataStream &stream) const
    {
        (void) stream;

        QString path = Globals->sdkPath + "/share/openbr/models/openbrcascades/" + model;
        QtUtils::touchDir(QDir(path));

        QString filename = path + "/cascade.xml";
        FileStorage fs(filename.toStdString(), FileStorage::WRITE);

        if (!fs.isOpened()) {
            qWarning("Unable to open file: %s", qPrintable(filename));
            return;
        }

        fs << FileStorage::getDefaultObjectName(filename.toStdString()) << "{";

        classifier->write(fs);

        fs << "}";
    }
};

BR_REGISTER(Transform, SlidingWindowTransform)

} // namespace br

#include "imgproc/slidingwindow.moc"
