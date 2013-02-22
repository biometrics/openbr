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
#include <openbr_plugin.h>

#include "core/opencvutils.h"
#include "core/resource.h"

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
        else                             qFatal("CascadeResourceMaker::CascadeResourceMaker invalid model.");
    }

private:
    CascadeClassifier *make() const
    {
        CascadeClassifier *cascade = new CascadeClassifier();
        if (!cascade->load(file.toStdString()))
            qFatal("CascadeResourceMaker::make failed to load: %s", qPrintable(file));
        return cascade;
    }
};


/*!
 * \ingroup transforms
 * \brief Wraps OpenCV cascade classifier
 * \author Josh Klontz \cite jklontz
 */
class CascadeTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(QString model READ get_model WRITE set_model RESET reset_model STORED false)
    Q_PROPERTY(int minSize READ get_minSize WRITE set_minSize RESET reset_minSize STORED false)
    BR_PROPERTY(QString, model, "FrontalFace")
    BR_PROPERTY(int, minSize, 64)

    Resource<CascadeClassifier> cascadeResource;

    void init()
    {
        cascadeResource.setResourceMaker(new CascadeResourceMaker(model));
    }

    void project(const Template &src, Template &dst) const
    {
        CascadeClassifier *cascade = cascadeResource.acquire();
        vector<Rect> rects;
        cascade->detectMultiScale(src, rects, 1.2, 5, src.file.getBool("enrollAll") ? 0 : CV_HAAR_FIND_BIGGEST_OBJECT, Size(minSize, minSize));
        cascadeResource.release(cascade);

        if (!src.file.getBool("enrollAll") && rects.empty())
            rects.push_back(Rect(0, 0, src.m().cols, src.m().rows));

        foreach (const Rect &rect, rects) {
            dst += src;
            dst.file.appendROI(OpenCVUtils::fromRect(rect));
        }
    }
};

BR_REGISTER(Transform, CascadeTransform)

} // namespace br

#include "cascade.moc"
