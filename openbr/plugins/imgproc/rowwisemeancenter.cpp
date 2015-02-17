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

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Remove the row-wise training set average.
 * \author Josh Klontz \cite jklontz
 */
class RowWiseMeanCenterTransform : public Transform
{
    Q_OBJECT
    Mat mean;

    void train(const TemplateList &data)
    {
        Mat m = OpenCVUtils::toMatByRow(data.data());
        mean = Mat(1, m.cols, m.type());
        for (int i=0; i<m.cols; i++)
            mean.col(i) = cv::mean(m.col(i));
    }

    void project(const Template &src, Template &dst) const
    {
        Mat m = src.m().clone();
        for (int i=0; i<m.rows; i++)
            m.row(i) -= mean;
        dst = m;
    }

    void store(QDataStream &stream) const
    {
        stream << mean;
    }

    void load(QDataStream &stream)
    {
        stream >> mean;
    }
};

BR_REGISTER(Transform, RowWiseMeanCenterTransform)

} // namespace br

#include "imgproc/rowwisemeancenter.moc"
