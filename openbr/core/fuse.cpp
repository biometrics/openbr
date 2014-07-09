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

#include <QList>
#include <QStringList>
#include "openbr/core/opencvutils.h"
#include <limits>
#include <vector>
#include <opencv2/core/core.hpp>
#include <openbr/openbr_plugin.h>

#include "openbr/core/bee.h"
#include "openbr/core/common.h"
#include "openbr/core/fuse.h"

using namespace cv;

static void normalizeMatrix(Mat &matrix, const Mat &mask, const QString &method)
{
    if (matrix.rows != mask.rows && matrix.cols != mask.cols)
        qFatal("Similarity matrix (%d, %d) and mask (%d, %d) size mismatch.", matrix.rows, matrix.cols, mask.rows, mask.cols);

    if (method == "None") return;

    QList<float> vals; vals.reserve(matrix.rows*matrix.cols);
    for (int i=0; i<matrix.rows; i++) {
        for (int j=0; j<matrix.cols; j++) {
            float val = matrix.at<float>(i,j);
            if ((mask.at<BEE::MaskValue>(i,j) == BEE::DontCare) ||
                (val == -std::numeric_limits<float>::max()) ||
                (val ==  std::numeric_limits<float>::max()))
                continue;
            vals.append(val);
        }
    }

    float min, max;
    double mean, stddev;
    Common::MinMax(vals, &min, &max);
    Common::MeanStdDev(vals, &mean, &stddev);

    if (method == "MinMax") {
        for (int i=0; i<matrix.rows; i++) {
            for (int j=0; j<matrix.cols; j++) {
                if (mask.at<BEE::MaskValue>(i,j) == BEE::DontCare) continue;
                float &val = matrix.at<float>(i,j);
                if      (val == -std::numeric_limits<float>::max()) val = 0;
                else if (val ==  std::numeric_limits<float>::max()) val = 1;
                else                                                     val = (val - min) / (max - min);
            }
        }
    } else if (method == "ZScore") {
        if (stddev == 0) qFatal("Stddev is 0.");
        for (int i=0; i<matrix.rows; i++) {
            for (int j=0; j<matrix.cols; j++) {
                if (mask.at<BEE::MaskValue>(i,j) == BEE::DontCare) continue;
                float &val = matrix.at<float>(i,j);
                if      (val == -std::numeric_limits<float>::max()) val = (min - mean) / stddev;
                else if (val ==  std::numeric_limits<float>::max()) val = (max - mean) / stddev;
                else                                                val = (val - mean) / stddev;
            }
        }
    } else {
        qFatal("Invalid normalization method %s.", qPrintable(method));
    }
}

void br::Fuse(const QStringList &inputSimmats, const QString &normalization, const QString &fusion, const QString &outputSimmat)
{
    qDebug("Fusing %d to %s", inputSimmats.size(), qPrintable(outputSimmat));

    QString target, query, previousTarget, previousQuery;
    QList<Mat> originalMatrices;
    foreach (const QString &simmat, inputSimmats) {
        originalMatrices.append(BEE::readMatrix(simmat,&target,&query));
        // Make we're fusing score matrices for the same set of targets and querys
        if (!previousTarget.isEmpty() && !previousQuery.isEmpty() && (previousTarget != target || previousQuery != query))
            qFatal("Target or query files are not the same across fused matrices.");
        previousTarget = target; previousQuery = query;
    }

    if ((originalMatrices.size() < 2) && (fusion != "None")) qFatal("Expected at least two similarity matrices.");
    if ((originalMatrices.size() > 1) && (fusion == "None")) qFatal("Expected exactly one similarity matrix.");

    const FileList targetFiles = TemplateList::fromGallery(target).files();
    const FileList queryFiles = TemplateList::fromGallery(query).files();

    int partition = 0;
    int crossValidate = Globals->crossValidate;
    Mat buffer = Mat::zeros(originalMatrices.last().size(),originalMatrices.last().type());

    do {
        QList<Mat> matrices;
        foreach (const Mat& matrix, originalMatrices)
            matrices.append(matrix.clone());

        Mat matrix_mask = BEE::makeMask(targetFiles,queryFiles,partition);
        for (int i=0; i<matrices.size(); i++)
            normalizeMatrix(matrices[i], matrix_mask, normalization);

        Mat fused;
        if (fusion == "Max") {
            max(matrices[0], matrices[1], fused);
            for (int i=2; i<matrices.size(); i++)
                max(fused, matrices[i], fused);
        } else if (fusion == "Min") {
            min(matrices[0], matrices[1], fused);
            for (int i=2; i<matrices.size(); i++)
                min(fused, matrices[i], fused);
        } else if (fusion.startsWith("Sum")) {
            QList<float> weights;
            QStringList words = fusion.right(fusion.size()-3).split(":", QString::SkipEmptyParts);
            if (words.size() == 0) {
                for (int k=0; k<matrices.size(); k++)
                    weights.append(1);
            } else if (words.size() == matrices.size()) {
                bool ok;
                for (int k=0; k<matrices.size(); k++) {
                    float weight = words[k].toFloat(&ok);
                    if (!ok) qFatal("Non-numerical weight %s.", qPrintable(words[k]));
                    weights.append(weight);
                }
            } else {
                qFatal("Number of weights does not match number of similarity matrices.");
            }

            addWeighted(matrices[0], weights[0], matrices[1], weights[1], 0, fused);
            for (int i=2; i<matrices.size(); i++)
                addWeighted(fused, 1, matrices[i], weights[i], 0, fused);
        } else if (fusion == "Replace") {
            if (matrices.size() != 2) qFatal("Replace fusion requires exactly two matrices.");
            fused = matrices.first().clone();
            matrices.last().copyTo(fused, matrix_mask != BEE::DontCare);
        } else if (fusion == "Difference") {
            if (matrices.size() != 2) qFatal("Difference fusion requires exactly two matrices.");
            subtract(matrices[0], matrices[1], fused);
        } else if (fusion == "None") {
            fused = matrices[0];
        } else {
            qFatal("Invalid fusion method %s.", qPrintable(fusion));
        }

        // We don't want to add scores where the mask says we shouldn't care
        Mat buffer_mask = Mat::ones(matrix_mask.size(),CV_8UC1);
        buffer_mask.setTo(0,matrix_mask==BEE::DontCare);

        add(buffer,fused,buffer,buffer_mask);

        partition++;

    } while (partition < crossValidate);

    BEE::writeMatrix(buffer, outputSimmat);
}
