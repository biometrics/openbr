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
#include <QProcess>
#include <iostream>
#include <QTemporaryFile>
#include <QPair>

using namespace cv;

namespace br
{

class MultiChannelCascadeTransform : public MetaTransform
{
    Q_OBJECT

    Q_PROPERTY(QString model READ get_model WRITE set_model RESET reset_model STORED false)
    Q_PROPERTY(int min_size READ get_min_size WRITE set_min_size RESET reset_min_size STORED false)
    Q_PROPERTY(bool ROCMode READ get_ROCMode WRITE set_ROCMode RESET reset_ROCMode STORED false)
    Q_PROPERTY(int model_size READ get_model_size WRITE set_model_size RESET reset_model_size STORED false)
    Q_PROPERTY(int num_features READ get_num_features WRITE set_num_features RESET reset_num_features STORED false)
    Q_PROPERTY(int num_classifiers READ get_num_classifiers WRITE set_num_classifiers RESET reset_num_classifiers STORED false)
    Q_PROPERTY(int bg_samples READ get_bg_samples WRITE set_bg_samples RESET reset_bg_samples STORED false)

    BR_PROPERTY(QString, model, "FrontalFace_MC")
    BR_PROPERTY(int, min_size, 36)
    BR_PROPERTY(bool, ROCMode, false)
    BR_PROPERTY(int, model_size, 36)
    BR_PROPERTY(int, num_features, std::numeric_limits<int>::max())
    BR_PROPERTY(int, num_classifiers, 2000)
    BR_PROPERTY(int, bg_samples, 10)

    void train(const TemplateList &data)
    {
        // Lets make some features!
        evaluator = FeatureEvaluator();
        evaluator.generateRandomFeatures(num_features, model_size, model_size, 1); //not providing args now

        // Now we need to build our training data
        QList<Mat> images; QList<int> labels;
        getImagesAndLabelsFromData(data, images, labels); // some logic to parse input data

        TrainingData td(evaluator, images, labels); // build the training data
        AdaBoost ada(td); // initialize boosting

        classifier = ada.train(num_classifiers); // get a decision tree with

        // Update feature evaluator to reflect learned features
        QList<Feature> classifier_features;
        getFeaturesFromTree(tree, classifier_features);
        evaluator = FeatureEvaluator(classifier_features);
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
            Mat img = t.m();
            // This is currently for a single scale, I need to research more about the best way
            // to implement multiscale detection.
            for (int row = 0; row < img.rows - model_size; row++) {
                for (int col = 0; col < img.cols - model_size; col++) {  
                    bool finishedCascade = true;
                    float confidence = 0.0f;
                    for (int i = 0; i < classifier.size(); i++) {
                        if (classifier[i].predict(evaluator.evaluate(img(Rect(row, col, model_size, model_size)), i)) == NEG) {
                            finishedCascade = false;
                            break;
                        }
                        confidence += classifier[i].weight;
                    }

                    if (finishedCascade || ROCMode) {
                        const QRectF rect(row, col, model_size, model_size);

                        Template u(t.file, img);
                        u.file.appendRect(rect);
                        u.file.set(model, rect);
                        u.file.set("Confidence", confidences[i]);
                        dst.append(u);
                    }
                }
            }
        }
    }

    void load(QDataStream &stream)
    {
        stream >> classifier;
        stream >> evaluator;
    }

    void store(QDataStream &stream)
    {
        stream << classifier;
        stream << evaluator;
    }

private:
    Cascade classifier;
    FeatureEvaluator evaluator;
};

BR_REGISTER(Transform, MultiChannelCascadeTransform)

} // namespace br

#include "multichannelcascade.moc"
