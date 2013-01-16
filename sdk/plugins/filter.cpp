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
#include <openbr_plugin.h>

#include "core/tanh_sse.h"

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Gamma correction
 * \author Josh Klontz \cite jklontz
 */
class GammaTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(float gamma READ get_gamma WRITE set_gamma RESET reset_gamma)
    BR_PROPERTY(float, gamma, 0.2)

    Mat lut;

    void init()
    {
        lut.create(256, 1, CV_32FC1);
        if (gamma == 0) for (int i=0; i<256; i++) lut.at<float>(i,0) = log((float)i);
        else            for (int i=0; i<256; i++) lut.at<float>(i,0) = pow(i, gamma);
    }

    void project(const Template &src, Template &dst) const
    {
        LUT(src, lut, dst);
    }
};

BR_REGISTER(Transform, GammaTransform)

/*!
 * \ingroup transforms
 * \brief Gaussian blur
 * \author Josh Klontz \cite jklontz
 */
class BlurTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(float sigma READ get_sigma WRITE set_sigma RESET reset_sigma STORED false)
    BR_PROPERTY(float, sigma, 1)

    void project(const Template &src, Template &dst) const
    {
        GaussianBlur(src, dst, Size(0,0), sigma);
    }
};

BR_REGISTER(Transform, BlurTransform)

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

/*!
 * \ingroup transforms
 * \brief Meyers, E.; Wolf, L.
 * “Using biologically inspired features for face processing,”
 * Int. Journal of Computer Vision, vol. 76, no. 1, pp 93–104, 2008.
 * \author Scott Klum \cite sklum
 */

class CSDNTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(float s READ get_s WRITE set_s RESET reset_s STORED false)
    BR_PROPERTY(int, s, 16)

    void project(const Template &src, Template &dst) const
    {
        if (src.m().channels() != 1) qFatal("ContrastEq::project expected single channel source matrix.");

        const int nRows = src.m().rows;
        const int nCols = src.m().cols;

        Mat m;
        src.m().convertTo(m, CV_32FC1);

        const int surround = s/2;

        for ( int i = 0; i < nRows; i++ )
        {
            for ( int j = 0; j < nCols; j++ )
            {
                int width = min( j+surround, nCols ) - max( 0, j-surround );
                int height = min( i+surround, nRows ) - max( 0, i-surround );

                Rect_<int> ROI(max(0, j-surround), max(0, i-surround), width, height);

                Scalar_<float> avg = mean(m(ROI));

                m.at<float>(i,j) = m.at<float>(i,j) - avg[0];
            }
        }

     m.convertTo(m, CV_8UC1);
     dst = m;

    }
};

BR_REGISTER(Transform, CSDNTransform)

/*!
 * \ingroup transforms
 * \brief Xiaoyang Tan; Triggs, B.;
 * "Enhanced Local Texture Feature Sets for Face Recognition Under Difficult Lighting Conditions,"
 * Image Processing, IEEE Transactions on , vol.19, no.6, pp.1635-1650, June 2010
 * \author Josh Klontz \cite jklontz
 */
class ContrastEqTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(float a READ get_a WRITE set_a RESET reset_a STORED false)
    Q_PROPERTY(float t READ get_t WRITE set_t RESET reset_t STORED false)
    BR_PROPERTY(float, a, 1)
    BR_PROPERTY(float, t, 0.1)

    void project(const Template &src, Template &dst) const
    {
        if (src.m().channels() != 1) qFatal("ContrastEq::project expected single channel source matrix.");

        // Stage 1
        Mat stage1;
        {
            Mat abs_dst;
            absdiff(src, Scalar(0), abs_dst);
            Mat pow_dst;
            pow(abs_dst, a, pow_dst);
            float denominator = pow((float)mean(pow_dst)[0], 1.f/a);
            src.m().convertTo(stage1, CV_32F, 1/denominator);
        }

        // Stage 2
        Mat stage2;
        {
            Mat abs_dst;
            absdiff(stage1, Scalar(0), abs_dst);
            Mat min_dst;
            min(abs_dst, t, min_dst);
            Mat pow_dst;
            pow(min_dst, a, pow_dst);
            float denominator = pow((float)mean(pow_dst)[0], 1.f/a);
            stage1.convertTo(stage2, CV_32F, 1/denominator);
        }

        // Hyperbolic tangent
        const int nRows = src.m().rows;
        const int nCols = src.m().cols;
        const float* p = (const float*)stage2.ptr();
        Mat m(nRows, nCols, CV_32FC1);
        for (int i=0; i<nRows; i++)
            for (int j=0; j<nCols; j++)
                m.at<float>(i, j) = fast_tanh(p[i*nCols+j]);
        dst = m;
    }
};

BR_REGISTER(Transform, ContrastEqTransform)

/*!
 * \ingroup transforms
 * \brief Raise each element to the specified power.
 * \author Josh Klontz \cite jklontz
 */
class PowTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(float power READ get_power WRITE set_power RESET reset_power STORED false)
    Q_PROPERTY(bool preserveSign READ get_preserveSign WRITE set_preserveSign RESET reset_preserveSign STORED false)
    BR_PROPERTY(float, power, 2)
    BR_PROPERTY(bool, preserveSign, false)

    void project(const Template &src, Template &dst) const
    {
        pow(src, power, dst);
        if (preserveSign) subtract(Scalar::all(0), dst, dst, src.m() < 0);
    }
};

BR_REGISTER(Transform, PowTransform)

} // namespace br

#include "filter.moc"
