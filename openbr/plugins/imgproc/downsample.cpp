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
 * \brief Downsample the rows and columns of a matrix.
 * \author Lacey Best-Rowden \cite lbestrowden
 */
class DownsampleTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int k READ get_k WRITE set_k RESET reset_k STORED false)
    BR_PROPERTY(int, k, 1)

    void project(const Template &src, Template &dst) const
    {
        if (src.m().channels() != 1)
            qFatal("Expected 1 channel matrix.");
        Mat input = src.m();
        Mat output(ceil((double)input.rows/k), ceil((double)input.cols/k), CV_32FC1);
        for (int r=0; r<output.rows; r++) {
            for (int c=0; c<output.cols; c++) {
                output.at<float>(r,c) = input.at<float>(r*k,c*k);
            }
        }
        dst.m() = output;
    }
};

BR_REGISTER(Transform, DownsampleTransform)

} // namespace br

#include "imgproc/downsample.moc"
