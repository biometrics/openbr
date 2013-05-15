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

#include <openbr/openbr_plugin.h>

#include "classify.h"
#include "openbr/core/qtutils.h"

// Helper struct for statistics accumulation
struct Counter
{
    float truePositive, falsePositive, falseNegative;
    Counter()
    {
        truePositive = 0;
        falsePositive = 0;
        falseNegative = 0;
    }
};

void br::EvalClassification(const QString &predictedInput, const QString &truthInput)
{
    qDebug("Evaluating classification of %s against %s", qPrintable(predictedInput), qPrintable(truthInput));

    TemplateList predicted(TemplateList::fromGallery(predictedInput));
    TemplateList truth(TemplateList::fromGallery(truthInput));
    if (predicted.size() != truth.size()) qFatal("Input size mismatch.");

    QHash<QString, Counter> counters;
    for (int i=0; i<predicted.size(); i++) {
        if (predicted[i].file.name != truth[i].file.name)
            qFatal("Input order mismatch.");

        // Typically these lists will be of length one, but this generalization allows measuring multi-class labeling accuracy.
        QString predictedSubject = predicted[i].file.get<QString>("Subject");
        QString trueSubject = truth[i].file.get<QString>("Subject");

        QStringList predictedSubjects(predictedSubject);
        QStringList trueSubjects(trueSubject);

        foreach (const QString &subject, trueSubjects.toVector() /* Hack to copy the list. */) {
            if (predictedSubjects.contains(subject)) {
                counters[subject].truePositive++;
                trueSubjects.removeOne(subject);
                predictedSubjects.removeOne(subject);
            } else {
                counters[subject].falseNegative++;
            }
        }

        for (int i=0; i<trueSubjects.size(); i++)
            foreach (const QString &subject, predictedSubjects)
                counters[subject].falsePositive += 1.f / predictedSubjects.size();
    }

    QSharedPointer<Output> output(Output::make("", FileList() << "Subject" << "Count" << "Precision" << "Recall" << "F-score", FileList(counters.size())));

    int tpc = 0;
    int fnc = 0;
    const QStringList keys = counters.keys();
    for (int i=0; i<counters.size(); i++) {
        const QString &subject = keys[i];
        const Counter &counter = counters[subject];
        tpc += counter.truePositive;
        fnc += counter.falseNegative;
        const int count = counter.truePositive + counter.falseNegative;
        const float precision = counter.truePositive / (float)(counter.truePositive + counter.falsePositive);
        const float recall = counter.truePositive / (float)(counter.truePositive + counter.falseNegative);
        const float fscore = 2 * precision * recall / (precision + recall);
        // problem -cao
        output->setRelative(File("", subject).get<int>("Label"), i, 0);
        output->setRelative(count, i, 1);
        output->setRelative(precision, i, 2);
        output->setRelative(recall, i, 3);
        output->setRelative(fscore, i, 4);
    }

    qDebug("Overall Accuracy = %f", (float)tpc / (float)(tpc + fnc));
}

void br::EvalRegression(const QString &predictedInput, const QString &truthInput)
{
    qDebug("Evaluating regression of %s against %s", qPrintable(predictedInput), qPrintable(truthInput));

    const TemplateList predicted(TemplateList::fromGallery(predictedInput));
    const TemplateList truth(TemplateList::fromGallery(truthInput));
    if (predicted.size() != truth.size()) qFatal("Input size mismatch.");

    float rmsError = 0;
    QStringList truthValues, predictedValues;
    for (int i=0; i<predicted.size(); i++) {
        if (predicted[i].file.name != truth[i].file.name)
            qFatal("Input order mismatch.");
        rmsError += pow(predicted[i].file.get<float>("Subject")-truth[i].file.get<float>("Subject"), 2.f);
        truthValues.append(QString::number(truth[i].file.get<float>("Subject")));
        predictedValues.append(QString::number(predicted[i].file.get<float>("Subject")));
    }

    QStringList rSource;
    rSource << "# Load libraries" << "library(ggplot2)" << "" << "# Set Data"
            << "Actual <- c(" + truthValues.join(",") + ")"
            << "Predicted <- c(" + predictedValues.join(",") + ")"
            << "data <- data.frame(Actual, Predicted)"
            << "" << "# Construct Plot" << "pdf(\"EvalRegression.pdf\")"
            << "print(qplot(Actual, Predicted, data=data, geom=\"jitter\", alpha=I(2/3)) + geom_abline(intercept=0, slope=1, color=\"forestgreen\", size=I(1)) + geom_smooth(size=I(1), color=\"mediumblue\") + theme_bw())"
            << "print(qplot(Actual, Predicted-Actual, data=data, geom=\"jitter\", alpha=I(2/3)) + geom_abline(intercept=0, slope=0, color=\"forestgreen\", size=I(1)) + geom_smooth(size=I(1), color=\"mediumblue\") + theme_bw())"
            << "dev.off()";


    QString rFile = "EvalRegression.R";
    QtUtils::writeFile(rFile, rSource);
    bool success = QtUtils::runRScript(rFile);
    if (success) QtUtils::showFile("EvalRegression.pdf");

    qDebug("RMS Error = %f", sqrt(rmsError/predicted.size()));
}
