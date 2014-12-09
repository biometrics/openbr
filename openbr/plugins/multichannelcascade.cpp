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
#include <opencv2/imgproc/imgproc.hpp>
#include "openbr_internal.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/resource.h"
#include "openbr/core/qtutils.h"
#include "openbr/learning/adaboost.h"
#include "openbr/learning/trainingdata.h"
#include "openbr/learning/feature.h"
#include "openbr/learning/cascade.h"
#include <QProcess>
#include <iostream>
#include <QTemporaryFile>
#include <QPair>

using namespace cv;

namespace br
{

class CascadeResourceMaker : public ResourceMaker<Cascade>
{
    QString file;

public:
    CascadeResourceMaker(const QString &model)
    {
        file = Globals->sdkPath + "/share/openbr/models/openbrcascades/"+model+"/cascade.xml";
        QFile touchFile(file);
        QtUtils::touchDir(touchFile);
    }

private:
    Cascade *make() const
    {
        return &Cascade::read(file);
    }
};

class MultiChannelCascadeTransform : public MetaTransform
{
    Q_OBJECT

    Q_PROPERTY(QString model READ get_model WRITE set_model RESET reset_model STORED false)
    Q_PROPERTY(int model_size READ get_model_size WRITE set_model_size RESET reset_model_size STORED false)
    Q_PROPERTY(bool ROCMode READ get_ROCMode WRITE set_ROCMode RESET reset_ROCMode STORED false)
    Q_PROPERTY(int num_features READ get_num_features WRITE set_num_features RESET reset_num_features STORED false)
    Q_PROPERTY(int num_classifiers READ get_num_classifiers WRITE set_num_classifiers RESET reset_num_classifiers STORED false)
    Q_PROPERTY(int bg_samples READ get_bg_samples WRITE set_bg_samples RESET reset_bg_samples STORED false)

    BR_PROPERTY(QString, model, "FrontalFace_MC")
    BR_PROPERTY(int, model_size, 36)
    BR_PROPERTY(bool, ROCMode, false)
    BR_PROPERTY(int, num_features, std::numeric_limits<int>::max())
    BR_PROPERTY(int, num_classifiers, 2000)
    BR_PROPERTY(int, bg_samples, 10)

    Resource<Cascade> cascadeResource;

    void init()
    {
        cascadeResource.setResourceMaker(new CascadeResourceMaker(model));
    }

    void train(const TemplateList &data)
    {
        // Lets make some features!
        FeatureEvaluator evaluator = FeatureEvaluator(num_features);
        evaluator.generateRandomFeatures(); //not providing args now

        // Now we need to build our training data
        QList<QList<Mat> > images; QList<int> labels;
        getImagesAndLabelsFromData(data, images, labels); // some logic to parse input data

        TrainingData td(evaluator, images, labels); // build the training data
        AdaBoost ada(td); // initialize boosting

        DecisionTree tree = ada.train(num_classifiers); // get a decision tree with

        // Update feature evaluator to reflect learned features
        QList<Feature> classifier_features;
        getFeaturesFromTree(tree, classifier_features);
        FeatureEvaluator best_evaluator = FeatureEvaluator(classifier_features);

        Cascade classifier(tree, best_evaluator);
        classifier.write(model);
    }

    void project(const Template &src, Template &dst) const
    {
        TemplateList temp;
        project(TemplateList() << src, temp);
        if (!temp.isEmpty()) dst = temp.first();
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        Cascade *cascade = cascadeResource.acquire();
        foreach (const Template &t, src) {
            Mat img = t.m();

            QList<QRectF> detections; QList<float> confidences;
            cascade->detect(img, detections, confidences);

            for (int i = 0; i < detections.size(); i++) {
                const QRectF rect = detections[i];

                Template u(t.file, img);
                u.file.appendRect(rect);
                u.file.set(model, rect);
                u.file.set("Confidence", confidences[i]);
                dst.append(u);
            }
        }
    }
};

BR_REGISTER(Transform, MultiChannelCascadeTransform)

} // namespace br

#include "cascade.moc"
