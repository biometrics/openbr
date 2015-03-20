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
 * \brief Prediction using only the red wavelength; magic numbers from jmp
 * \author E. Taborsky \cite mmtaborsky
 */
class RedLinearRegressionTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        Mat m; src[0].convertTo(m, CV_32F); assert(m.isContinuous() && (m.channels() == 1));

        const float rmult = .6533673;
        const float add = 41.268;

        Mat dst1(m.size(), CV_32F);
        int rows = m.rows;
        int cols = m.cols;

        const float *rsrc = (const float*) m.ptr();
        float *p = (float*)dst1.ptr();

        for (int r = 0; r < rows; r++){
            for (int c = 0; c < cols; c++){
                int index = r*cols+c;
                const float rval = rsrc[index];
                p[index] = rval*rmult+add;
            }
        }
        dst = dst1;
    }
};

BR_REGISTER(Transform, RedLinearRegressionTransform)

} // namespace br

#include "imgproc/redlinearregression.moc"
