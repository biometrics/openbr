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

using namespace std;
using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Use biologically inspired features for face processing (see paper).
 * \br_paper Meyers, E.; Wolf, L.
 *           “Using biologically inspired features for face processing,”
 *           Int. Journal of Computer Vision, vol. 76, no. 1, pp 93–104, 2008.
 * \author Scott Klum \cite sklum
 */

class CSDNTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(float s READ get_s WRITE set_s RESET reset_s STORED false)
    BR_PROPERTY(int, s, 16)

    void project(const Template &src, Template &dst) const
    {
        if (src.m().channels() != 1) qFatal("Expected single channel source matrix.");

        const int nRows = src.m().rows;
        const int nCols = src.m().cols;

        Mat m;
        src.m().convertTo(m, CV_32FC1);

        const int surround = s/2;

        for ( int i = 0; i < nRows; i++ ) {
            for ( int j = 0; j < nCols; j++ ) {
                int width = min( j+surround, nCols ) - max( 0, j-surround );
                int height = min( i+surround, nRows ) - max( 0, i-surround );

                Rect_<int> ROI(max(0, j-surround), max(0, i-surround), width, height);

                Scalar_<float> avg = mean(m(ROI));

                m.at<float>(i,j) = m.at<float>(i,j) - avg[0];
            }
        }

        dst = m;

    }
};

BR_REGISTER(Transform, CSDNTransform)

} // namespace br

#include "imgproc/csdn.moc"
