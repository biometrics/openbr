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

#include <Eigen/Dense>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace Eigen;
using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Consolidate redundant/overlapping detections.
 * \author Brendan Klare \cite bklare
 */
class ConsolidateDetectionsTransform : public UntrainableMetadataTransform
{
    Q_OBJECT

    void projectMetadata(const File &src, File &dst) const
    {
        dst = src;
        if (!dst.contains("Confidences"))
            return;

        //Compute overlap between rectangles and create discrete Laplacian matrix
        QList<Rect> rects = OpenCVUtils::toRects(src.rects());
        int n = rects.size();
        if (n == 0)
            return;
        MatrixXf laplace(n,n);
        for (int i = 0; i < n; i++) {
            laplace(i,i) = 0;
        }
        for (int i = 0; i < n; i++){
            for (int j = i + 1; j < n; j++) {
                float overlap = (float)((rects[i] & rects[j]).area()) / (float)max(rects[i].area(), rects[j].area());
                if (overlap > 0.5) {
                    laplace(i,j) = -1.0;
                    laplace(j,i) = -1.0;
                    laplace(i,i) = laplace(i,i) + 1.0;
                    laplace(j,j) = laplace(j,j) + 1.0;
                } else {
                    laplace(i,j) = 0;
                    laplace(j,i) = 0;
                }
            }
        }

        // Compute eigendecomposition
        SelfAdjointEigenSolver<Eigen::MatrixXf> eSolver(laplace);
        MatrixXf allEVals = eSolver.eigenvalues();
        MatrixXf allEVecs = eSolver.eigenvectors();

        //Keep eigenvectors with zero eigenvalues
        int nRegions = 0;
        for (int i = 0; i < n; i++) {
            if (fabs(allEVals(i)) < 1e-4) {
                nRegions++;
            }
        }
        MatrixXf regionVecs(n, nRegions);
        for (int i = 0, cnt = 0; i < n; i++) {
            if (fabs(allEVals(i)) < 1e-4)
                regionVecs.col(cnt++) = allEVecs.col(i);
        }

        //Determine membership for each consolidated location
        // and compute average of regions. This is determined by
        // finding which eigenvector has the highest magnitude for
        // each input dimension. Each input dimension corresponds to
        // one of the input rect region. Thus, each eigenvector represents
        // a set of overlaping regions.
        float *midX = new float[nRegions];
        float *midY = new float[nRegions];
        float *avgWidth = new float[nRegions];
        float *avgHeight = new float[nRegions];
        float *confs = new float[nRegions];
        int *cnts = new int[nRegions];
        int mx;
        int mxIdx;
        for (int i = 0 ; i < nRegions; i++) {
            midX[i] = 0;
            midY[i] = 0;
            avgWidth[i] = 0;
            avgHeight[i] = 0;
            confs[i] = 0;
            cnts[i] = 0;
        }

        QList<float> confidences = dst.getList<float>("Confidences");
        for (int i = 0; i < n; i++) {
            mx = 0.0;
            mxIdx = -1;

            for (int j = 0; j < nRegions; j++) {
                if (fabs(regionVecs(i,j)) > mx) {
                    mx = fabs(regionVecs(i,j));
                    mxIdx = j;
                }
            }

            Rect curRect = rects[i];
            midX[mxIdx] += ((float)curRect.x + (float)curRect.width  / 2.0);
            midY[mxIdx] += ((float)curRect.y + (float)curRect.height / 2.0);
            avgWidth[mxIdx]  += (float) curRect.width;
            avgHeight[mxIdx] += (float) curRect.height;
            confs[mxIdx] += confidences[i];
            cnts[mxIdx]++;
        }

        QList<Rect> consolidatedRects;
        QList<float> consolidatedConfidences;
        for (int i = 0; i < nRegions; i++) {
            float cntF = (float) cnts[i];
            if (cntF > 0) {
                int x = qRound((midX[i] / cntF) - (avgWidth[i] / cntF) / 2.0);
                int y = qRound((midY[i] / cntF) - (avgHeight[i] / cntF) / 2.0);
                int w = qRound(avgWidth[i] / cntF);
                int h = qRound(avgHeight[i] / cntF);
                consolidatedRects.append(Rect(x,y,w,h));
                consolidatedConfidences.append(confs[i] / cntF);
            }
        }

        delete [] midX;
        delete [] midY;
        delete [] avgWidth;
        delete [] avgHeight;
        delete [] confs;
        delete [] cnts;

        dst.setRects(consolidatedRects);
        dst.setList<float>("Confidences", consolidatedConfidences);
    }
};

BR_REGISTER(Transform, ConsolidateDetectionsTransform)

} // namespace br

#include "metadata/consolidatedetections.moc"
