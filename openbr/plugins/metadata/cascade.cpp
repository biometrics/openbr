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
#include <QProcess>
#include <QTemporaryFile>
#include <opencv2/objdetect/objdetect.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>
#include <openbr/core/resource.h>
#include <openbr/core/qtutils.h>
#include <openbr/core/cascade.h>

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
        else {
            // Create a directory for trainable cascades
            file += "openbrcascades/"+model+"/cascade.xml";
            QFile touchFile(file);
            QtUtils::touchDir(touchFile);
        }                             
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
 * \author David Crouse \cite dgcrouse
 */
class CascadeTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString model READ get_model WRITE set_model RESET reset_model STORED false)
    Q_PROPERTY(int minSize READ get_minSize WRITE set_minSize RESET reset_minSize STORED false)
    Q_PROPERTY(int minNeighbors READ get_minNeighbors WRITE set_minNeighbors RESET reset_minNeighbors STORED false)
    Q_PROPERTY(bool ROCMode READ get_ROCMode WRITE set_ROCMode RESET reset_ROCMode STORED false)
    
    // Training parameters
    Q_PROPERTY(QString vecFile READ get_vecFile WRITE set_vecFile RESET reset_vecFile STORED false)
    Q_PROPERTY(QString negFile READ get_negFile WRITE set_negFile RESET reset_negFile STORED false)
    Q_PROPERTY(int numPos READ get_numPos WRITE set_numPos RESET reset_numPos STORED false)
    Q_PROPERTY(int numNeg READ get_numNeg WRITE set_numNeg RESET reset_numNeg STORED false)

    BR_PROPERTY(QString, model, "FrontalFace")
    BR_PROPERTY(int, minSize, 64)
    BR_PROPERTY(int, minNeighbors, 5)
    BR_PROPERTY(bool, ROCMode, false)                 

    BR_PROPERTY(QString, vecFile, "data.vec")
    BR_PROPERTY(QString, negFile, "neg.txt")
    BR_PROPERTY(int, numPos, 1000)
    BR_PROPERTY(int, numNeg, 1000)

    Resource<CascadeClassifier> cascadeResource;

    void init()
    {
        cascadeResource.setResourceMaker(new CascadeResourceMaker(model));
        if (model == "Ear" || model == "Eye" || model == "FrontalFace" || model == "ProfileFace")
            this->trainable = false;
    }
    
    // Train transform
    void train(const TemplateList& data)
    {
        (void)data;

        BrCascadeClassifier classifier;

        CascadeParams cascadeParams(CascadeParams::BOOST, FeatureParams::LBP);
        CascadeBoostParams stageParams(CvBoost::GENTLE, 0.999, 0.5, 0.95, 1, 200);
        LBPFeatureParams featureParams;

        QString cascadeDir = Globals->sdkPath + "/share/openbr/models/openbrcascades/" + model;
        classifier.train(cascadeDir.toStdString(),
                         vecFile.toStdString(), negFile.toStdString(),
                         numPos, numNeg, 1024, 1024, 17,
                         cascadeParams, featureParams, stageParams);
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
                if (ROCMode) cascade->detectMultiScale(m, rects, rejectLevels, levelWeights, 1.2, minNeighbors, (enrollAll ? 0 : CASCADE_FIND_BIGGEST_OBJECT) | CASCADE_SCALE_IMAGE, Size(minSize, minSize), Size(), true);
                else         cascade->detectMultiScale(m, rects, 1.2, minNeighbors, enrollAll ? 0 : CASCADE_FIND_BIGGEST_OBJECT, Size(minSize, minSize));

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

#include "metadata/cascade.moc"
