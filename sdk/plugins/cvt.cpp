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

#include "core/opencvutils.h"

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Colorspace conversion
 * \author Josh Klontz \cite jklontz
 */
class CvtTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_ENUMS(Code)
    Q_PROPERTY(Code code READ get_code WRITE set_code RESET reset_code STORED false)
    Q_PROPERTY(int channel READ get_channel WRITE set_channel RESET reset_channel STORED false)

public:
    /*!< */
    enum Code { Gray = CV_BGR2GRAY,
                RGBGray = CV_RGB2GRAY,
                HLS = CV_BGR2HLS,
                HSV = CV_BGR2HSV,
                Lab = CV_BGR2Lab,
                Luv = CV_BGR2Luv,
                RGB = CV_BGR2RGB,
                XYZ = CV_BGR2XYZ,
                YCrCb = CV_BGR2YCrCb };

private:
    BR_PROPERTY(Code, code, Gray)
    BR_PROPERTY(int, channel, -1)

    void project(const Template &src, Template &dst) const
    {
        if (src.m().channels() > 1) cvtColor(src, dst, code);
        else                        dst = src;

        if (channel != -1) {
            std::vector<Mat> mv;
            split(dst, mv);
            dst = mv[channel % (int)mv.size()];
        }
    }
};

BR_REGISTER(Transform, CvtTransform)

/*!
 * \ingroup transforms
 * \brief Convert to floating point format.
 * \author Josh Klontz \cite jklontz
 */
class CvtFloatTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        src.m().convertTo(dst, CV_32F);
    }
};

BR_REGISTER(Transform, CvtFloatTransform)

/*!
 * \ingroup transforms
 * \brief Convert to uchar format
 * \author Josh Klontz \cite jklontz
 */
class CvtUCharTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        OpenCVUtils::cvtUChar(src, dst);
    }
};

BR_REGISTER(Transform, CvtUCharTransform)

/*!
 * \ingroup transforms
 * \brief Split a multi-channel matrix into several single-channel matrices.
 * \author Josh Klontz \cite jklontz
 */
class SplitChannelsTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        std::vector<Mat> mv;
        split(src, mv);
        foreach (const Mat &m, mv)
            dst += m;
    }
};

BR_REGISTER(Transform, SplitChannelsTransform)

/*!
 * \ingroup transforms
 * \brief Enforce the matrix has a certain number of channels by adding or removing channels.
 * \author Josh Klontz \cite jklontz
 */
class EnsureChannelsTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int n READ get_n WRITE set_n RESET reset_n STORED false)
    BR_PROPERTY(int, n, 1)

    void project(const Template &src, Template &dst) const
    {
        if (src.m().channels() == n) {
            dst = src;
        } else {
            std::vector<Mat> mv;
            split(src, mv);

            // Add extra channels
            while ((int)mv.size() < n) {
                for (int i=0; i<src.m().channels(); i++) {
                    mv.push_back(mv[i]);
                    if ((int)mv.size() == n)
                        break;
                }
            }

            // Remove extra channels
            while ((int)mv.size() > n)
                mv.pop_back();

            merge(mv, dst);
        }
    }
};

BR_REGISTER(Transform, EnsureChannelsTransform)

/*!
 * \ingroup transforms
 * \brief Normalized RG color space.
 * \author Josh Klontz \cite jklontz
 */
class RGTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        if (src.m().type() != CV_8UC3)
            qFatal("Expected CV_8UC3 images.");

        const Mat &m = src.m();
        Mat R(m.size(), CV_8UC1); // R / (R+G+B)
        Mat G(m.size(), CV_8UC1); // G / (R+G+B)

        for (int i=0; i<m.rows; i++)
            for (int j=0; j<m.cols; j++) {
                Vec3b v = m.at<Vec3b>(i,j);
                uchar& b = v[0];
                uchar& g = v[1];
                uchar& r = v[2];

                R.at<uchar>(i, j) = saturate_cast<uchar>(255.0*r/(r+g+b));
                G.at<uchar>(i, j) = saturate_cast<uchar>(255.0*g/(r+g+b));
            }

        dst.append(R);
        dst.append(G);
    }
};

BR_REGISTER(Transform, RGTransform)

/*!
 * \ingroup transforms
 * \brief dst = a*src+b
 * \author Josh Klontz \cite jklontz
 */
class MAddTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(double a READ get_a WRITE set_a RESET reset_a STORED false)
    Q_PROPERTY(double b READ get_b WRITE set_b RESET reset_b STORED false)
    BR_PROPERTY(double, a, 1)
    BR_PROPERTY(double, b, 0)

    void project(const Template &src, Template &dst) const
    {
        src.m().convertTo(dst.m(), src.m().depth(), a, b);
    }
};

BR_REGISTER(Transform, MAddTransform)

} // namespace br

#include "cvt.moc"
