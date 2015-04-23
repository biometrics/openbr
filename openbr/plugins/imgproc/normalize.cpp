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
 * \brief Normalize matrix to unit length
 * \author Josh Klontz \cite jklontz
 * \br_property enum NormType Values are: [NORM_INF, NORM_L1, NORM_L2, NORM_MINMAX]
 * \br_property bool ByRow If true normalize each row independently otherwise normalize the entire matrix.
 * \br_property int alpha Lower bound if using NORM_MINMAX. Value to normalize to otherwise.
 * \br_property int beta Upper bound if using NORM_MINMAX. Not used otherwise.
 * \br_property bool squareRoot If true compute the signed square root of the output after normalization.
 */
class NormalizeTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_ENUMS(NormType)
    Q_PROPERTY(NormType normType READ get_normType WRITE set_normType RESET reset_normType STORED false)

    Q_PROPERTY(bool ByRow READ get_ByRow WRITE set_ByRow RESET reset_ByRow STORED false)
    BR_PROPERTY(bool, ByRow, false)
    Q_PROPERTY(int alpha READ get_alpha WRITE set_alpha RESET reset_alpha STORED false)
    BR_PROPERTY(int, alpha, 1)
    Q_PROPERTY(int beta READ get_beta WRITE set_beta RESET reset_beta STORED false)
    BR_PROPERTY(int, beta, 0)
    Q_PROPERTY(bool squareRoot READ get_squareRoot WRITE set_squareRoot RESET reset_squareRoot STORED false)
    BR_PROPERTY(bool, squareRoot, false)

public:
    /*!< */
    enum NormType { Inf = NORM_INF,
                    L1 = NORM_L1,
                    L2 = NORM_L2,
                    Range = NORM_MINMAX };

private:
    BR_PROPERTY(NormType, normType, L2)

    static void signedSquareRoot(Mat &m)
    {
        for (int i=0; i<m.rows; i++)
            for (int j=0; j<m.cols; j++) {
                float &val = m.at<float>(i, j);
                val = sqrtf(fabsf(val)) * (val >= 0 ? 1 : -1);
            }
    }

    void project(const Template &src, Template &dst) const
    {
        if (!ByRow) {
            normalize(src, dst, alpha, beta, normType, CV_32F);
            if (squareRoot)
                signedSquareRoot(dst);
        }

        else {
            dst = src;
            for (int i=0; i<dst.m().rows; i++) {
                Mat temp;
                normalize(dst.m().row(i), temp, alpha, beta, normType);
                if (squareRoot)
                    signedSquareRoot(temp);
                temp.copyTo(dst.m().row(i));
            }
        }

    }
};

BR_REGISTER(Transform, NormalizeTransform)

} // namespace br

#include "imgproc/normalize.moc"
