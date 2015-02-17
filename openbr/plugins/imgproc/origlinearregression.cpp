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

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Prediction with magic numbers from jmp; must get input as blue;green;red
 * \author E. Taborsky \cite mmtaborsky
 */
class OrigLinearRegressionTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        if (src.size() != 3) qFatal("Expected exactly three source images, got %d.", src.size());
        Mat m1; src[0].convertTo(m1, CV_32F); assert(m1.isContinuous() && (m1.channels() == 1));
        Mat m2; src[1].convertTo(m2, CV_32F); assert(m2.isContinuous() && (m2.channels() == 1));
        Mat m3; src[2].convertTo(m3, CV_32F); assert(m3.isContinuous() && (m3.channels() == 1));

        const float rmult = .809911, gmult = -.09625, bmult = -.020115, add = 35.78;

        Mat dstmat(m1.size(), CV_32F);

        int rows = m1.rows;
        int cols = m1.cols;

        const float *pb = (const float*) m1.ptr(), *pg = (const float*) m2.ptr(), *pr = (const float*) m3.ptr();
        float *pd = (float*)dstmat.ptr();
        for (int r = 0; r < rows; r++){
            for (int c = 0; c < cols; c++){
                int index = r*cols+c;
                const float bval = pb[index], gval = pg[index], rval = pr[index];
                pd[index] = bval*bmult+ gval*gmult+rval*rmult+add;
            }
        }
        dst = dstmat;
    }
};

BR_REGISTER(Transform, OrigLinearRegressionTransform)

} // namespace br

#include "imgproc/origlinearregression.moc"
