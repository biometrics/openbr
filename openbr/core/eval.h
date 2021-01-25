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

#ifndef BR_EVAL_H
#define BR_EVAL_H

#include <QList>
#include <QString>
#include "openbr/openbr_plugin.h"

namespace br
{
    float Evaluate(const QString &simmat, const QString &mask = "", const File &csv = "", unsigned int matches = 0); // Returns TAR @ FAR = 0.001
    float Evaluate(const cv::Mat &scores, const FileList &target, const FileList &query, const File &csv = "", int parition = 0);
    float Evaluate(const cv::Mat &scores, const cv::Mat &masks, const File &csv = "", const QString &target = "", const QString &query = "", unsigned int matches = 0);
    void assertEval(const QString &simmat, const QString &mask, float accuracy); // Check to see if -eval achieves a given TAR @ FAR = 0.001
    float InplaceEval(const QString & simmat, const QString & target, const QString & query, const QString & csv = "");

    void EvalClassification(const QString &predictedGallery, const QString &truthGallery, QString predictedProperty = "", QString truthProperty = "");
    float EvalDetection(const QString &predictedGallery, const QString &truthGallery, const QString &csv = "", bool normalize = false, int minSize = 0, int maxSize = 0, float relativeMinSize = 0); // Return average overlap
    float EvalLandmarking(const QString &predictedGallery, const QString &truthGallery, const QString &csv = "", int normalizationIndexA = 0, int normalizationIndexB = 1, int sampleIndex = 0, int totalExamples = 5); // Return average error
    void EvalRegression(const QString &predictedGallery, const QString &truthGallery, QString predictedProperty = "", QString truthProperty = "");
    void EvalKNN(const QString &knnGraph, const QString &knnTruth, const QString &csv = "");
    void EvalEER(const QString &predictedXML, const QString gt_property = "", const QString distribution_property = "", const QString &csv = "");
    struct Candidate
    {
        size_t index;
        float similarity;

        Candidate() {}
        Candidate(size_t index_, float similarity_)
            : index(index_), similarity(similarity_) {}
    };
}

#endif // BR_EVAL_H

