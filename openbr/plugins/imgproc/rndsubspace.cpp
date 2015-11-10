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
 * \brief Generates a random subspace.
 * \author Josh Klontz \cite jklontz
 */
class RndSubspaceTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(float fraction READ get_fraction WRITE set_fraction RESET reset_fraction STORED false)
    Q_PROPERTY(bool weighted READ get_weighted WRITE set_weighted RESET reset_weighted STORED false)
    BR_PROPERTY(float, fraction, 0.5)
    BR_PROPERTY(bool, weighted, false)

    QList<Mat> maps;

    void train(const TemplateList &data)
    {
        // While RndSubspace transform could be independent, making it a MetaTransform is a workaround
        // for parallel random number generation being difficult on Windows due to thread-local
        // RNG states all initialized with the same seed and thus producing the same sequence of random numbers.
        for (int index=0; index<data.first().size(); index++) {
            const int cols = data.first()[index].cols;
            const int size = data.first()[index].rows * cols;
            QList<float> weights; weights.reserve(size);
            if (weighted) {
                Mat flatData = OpenCVUtils::toMat(data.data());
                for (int i=0; i<size; i++) {
                    Scalar mean, stddev;
                    cv::meanStdDev(flatData.col(i), mean, stddev);
                    weights.append(pow(stddev[0],2.0));
                }
            } else {
                for (int i=0; i<size; i++)
                    weights.append(1);
            }
            const int dimsOut = std::max(int(weights.size()*fraction), 1);

            QList<int> sample = Common::RandSample(dimsOut, weights);
            Mat xMap(1, dimsOut, CV_16SC1);
            Mat yMap(1, dimsOut, CV_16SC1);
            for (int j=0; j<dimsOut; j++) {
                int index = sample[j];
                xMap.at<short>(0,j) = index % cols;
                yMap.at<short>(0,j) = index / cols;
            }
            std::vector<Mat> mv;
            mv.push_back(xMap);
            mv.push_back(yMap);

            Mat map;
            merge(mv, map);
            maps.append(map);
        }
    }

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;
        for (int i=0; i<src.size(); i++) {
            Mat m;
            remap(src, m, maps[i], Mat(), INTER_NEAREST);
            dst.append(m);
        }
    }

    void store(QDataStream &stream) const
    {
        stream << maps;
    }

    void load(QDataStream &stream)
    {
        stream >> maps;
    }
};

BR_REGISTER(Transform, RndSubspaceTransform)

} // namespace br

#include "imgproc/rndsubspace.moc"
