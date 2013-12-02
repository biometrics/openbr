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

#include <opencv2/objdetect/objdetect.hpp>
//#include <opencv2/objdetect/objdetect_c.h>
#include "openbr_internal.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/resource.h"

using namespace cv;

namespace br
{

class CascadeResourceMaker : public ResourceMaker<CascadeClassifier>
{
    QString file;

public:
    CascadeResourceMaker(const QString &model)
    {
        file = Globals->sdkPath + "/share/openbr/models/";
        if      (model == "Ear")         file += "haarcascades/haarcascade_ear.xml";
        else if (model == "Eye")         file += "haarcascades/haarcascade_eye_tree_eyeglasses.xml";
        else if (model == "FrontalFace") file += "haarcascades/haarcascade_frontalface_alt2.xml";
        else if (model == "ProfileFace") file += "haarcascades/haarcascade_profileface.xml";
        else                             qFatal("Invalid model.");
    }

private:
    CascadeClassifier *make() const
    {
        CascadeClassifier *cascade = new CascadeClassifier();
        if (!cascade->load(file.toStdString()))
            qFatal("Failed to load: %s", qPrintable(file));
        return cascade;
    }
};

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV cascade classifier
 * \author Josh Klontz \cite jklontz
 */
class CascadeTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString model READ get_model WRITE set_model RESET reset_model STORED false)
    Q_PROPERTY(int minSize READ get_minSize WRITE set_minSize RESET reset_minSize STORED false)
    Q_PROPERTY(bool ROCMode READ get_ROCMode WRITE set_ROCMode RESET reset_ROCMode STORED false)
    BR_PROPERTY(QString, model, "FrontalFace")
    BR_PROPERTY(int, minSize, 64)
    BR_PROPERTY(bool, ROCMode, false)

    Resource<CascadeClassifier> cascadeResource;

    void init()
    {
        cascadeResource.setResourceMaker(new CascadeResourceMaker(model));
    }

    void project(const Template &src, Template &dst) const
    {
        TemplateList temp;
        project(TemplateList() << src, temp);
        if (!temp.isEmpty()) dst = temp.first();
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        CascadeClassifier *cascade = cascadeResource.acquire();
        foreach (const Template &t, src) {
            const bool enrollAll = t.file.getBool("enrollAll");

            for (int i=0; i<t.size(); i++) {
                const Mat &m = t[i];
                std::vector<Rect> rects;
                std::vector<int> rejectLevels;
                std::vector<double> levelWeights;
                if (ROCMode) cascade->detectMultiScale(m, rects, rejectLevels, levelWeights, 1.2, 5, (enrollAll ? 0 : CV_HAAR_FIND_BIGGEST_OBJECT) | CASCADE_SCALE_IMAGE, Size(minSize, minSize), Size(), true);
                else         cascade->detectMultiScale(m, rects, 1.2, 5, enrollAll ? 0 : CASCADE_FIND_BIGGEST_OBJECT, Size(minSize, minSize));

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

        cascadeResource.release(cascade);
    }

    // TODO: Remove this code when ready to break binary compatibility
    void store(QDataStream &stream) const
    {
        int size = 1;
        stream << size;
    }

    void load(QDataStream &stream)
    {
        int size;
        stream >> size;
    }
};

BR_REGISTER(Transform, CascadeTransform)

} // namespace br

#include "cascade.moc"
