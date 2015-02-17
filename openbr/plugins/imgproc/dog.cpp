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

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Difference of gaussians
 * \author Josh Klontz \cite jklontz
 */
class DoGTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(float sigma0 READ get_sigma0 WRITE set_sigma0 RESET reset_sigma0 STORED false)
    Q_PROPERTY(float sigma1 READ get_sigma1 WRITE set_sigma1 RESET reset_sigma1 STORED false)
    BR_PROPERTY(float, sigma0, 1)
    BR_PROPERTY(float, sigma1, 2)

    Size ksize0, ksize1;

    static Size getKernelSize(double sigma)
    {
        // Inverts OpenCV's conversion from kernel size to sigma:
        // sigma = ((ksize-1)*0.5 - 1)*0.3 + 0.8
        // See documentation for cv::getGaussianKernel()
        int ksize = ((sigma - 0.8) / 0.3 + 1) * 2 + 1;
        if (ksize % 2 == 0) ksize++;
        return Size(ksize, ksize);
    }

    void init()
    {
        ksize0 = getKernelSize(sigma0);
        ksize1 = getKernelSize(sigma1);
    }

    void project(const Template &src, Template &dst) const
    {
        Mat g0, g1;
        GaussianBlur(src, g0, ksize0, 0);
        GaussianBlur(src, g1, ksize1, 0);
        subtract(g0, g1, dst);
    }
};

BR_REGISTER(Transform, DoGTransform)

} // namespace br

#include "imgproc/dog.moc"
