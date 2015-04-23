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

#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/common.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Face Recognition Using Early Biologically Inspired Features
 * \br_paper Li, Min, et al.
 *           "Face recognition using early biologically inspired features."
 *           Biometrics: Theory, Applications and Systems (BTAS), 2013 IEEE Sixth International Conference on. IEEE, 2013.
 * \author Josh Klontz \cite jklontz
 * \br_property int N The number of scales. Default is 6.
 * \br_property int M The number of orientations between 0 and pi. Default is 9.
 */
class EBIFTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int N READ get_N WRITE set_N RESET reset_N STORED false) // scales
    Q_PROPERTY(int M READ get_M WRITE set_M RESET reset_M STORED false) // orientations
    BR_PROPERTY(int, N, 6)
    BR_PROPERTY(int, M, 9)

    QList<Transform*> orientations;

    void init()
    {
        for (int m=0; m<M; m++)
            orientations.append(make(QString("Gabor(%1,%2,%3,%4,%5,Phase)+Abs").arg(
                                     QString::number(5), // lambda = 5 (just one wavelength)
                                     QString::number(CV_PI*m/M), // M orientations between 0 and pi
                                     QString::number(0), // psi = 0 (no offset)
                                     QString::number(3), // sigma = 3 (just one width)
                                     QString::number(1) // gamma = 1 (no skew)
                                     )));
    }

    void project(const Template &src, Template &dst) const
    {
        // Compute the image pyramid
        Template scales;
        float scaleFactor = 1;
        for (int n=0; n<N; n++) {
            Mat scale;
            const int width = src.m().cols * scaleFactor;
            const int height = src.m().rows * scaleFactor;
            resize(src, scale, Size(width, height));
            scale.convertTo(scale, CV_32F);
            scales.append(scale);
            scaleFactor /= sqrt(2.f);
        }

        // Mean and standard deviation pooling
        QList< QList<float> > features;
        foreach (Transform *orientation, orientations) {
            // Compute the reponse wavelet response
            Template response;
            orientation->project(scales, response);

            // Pool for each two adjacent features
            QList<float> orientedFeatures;
            for (int i=0; i<N-1; i++)
                orientedFeatures.append(pool(response[i], response[i+1]));
            features.append(orientedFeatures);
        }

        // L2 normalization across orientations
        for (int i=0; i<features.first().size(); i++) {
            float squaredSum = 0;
            for (int m=0; m<M; m++) {
                const float val = features[m][i];
                squaredSum += val * val;
            }
            const float norm = 1/sqrt(squaredSum + 0.001 /* Avoid division by zero */);
            for (int m=0; m<M; m++)
                features[m][i] *= norm;
        }

        // Group features by location (not done in paper)
        for (int i=0; i<features.first().size(); i+=2) {
            QList<float> localFeatures; localFeatures.reserve(2*M);
            for (int m=0; m<M; m++) {
                localFeatures.append(features[m][i]); // mean
                localFeatures.append(features[m][i+1]); // standard deviation
            }
            dst += OpenCVUtils::toMat(localFeatures);
        }
    }

    QList<float> pool(const Mat &bottom, const Mat &top) const
    {
        QList<float> features;
        for (int i=0; i<=top.rows-3; i+=3) {
            for (int j=0; j<=top.cols-3; j+=3) {
                QList<float> vals; vals.reserve(3*3 + 4*4);

                // Top values
                for (int k=0; k<3; k++) {
                    const float *data = top.ptr<float>(i+k, j);
                    for (int l=0; l<3; l++)
                        vals.append(data[l]);
                }

                // Bottom values
                for (int k=0; k<4; k++) {
                    const float *data = bottom.ptr<float>(4*i/3+k, 4*j/3);
                    for (int l=0; l<4; l++)
                        vals.append(data[l]);
                }

                double mean, stddev;
                Common::MeanStdDev(vals, &mean, &stddev);
                features.append(mean);
                features.append(stddev);
            }
        }

        return features;
    }
};

BR_REGISTER(Transform, EBIFTransform)

} // namespace br

#include "classification/ebif.moc"
