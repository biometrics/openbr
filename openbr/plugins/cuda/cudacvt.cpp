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

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


using namespace cv;

namespace br { namespace cuda{
  void cudacvt_wrapper(void* srcPtr, void**dstPtr, int rows, int cols);
}}

namespace br
{

/*!
 * \ingroup transforms
 * \brief Colorspace conversion.
 * \author Li Li \cite Josh Klontz \cite jklontz
 */
class CUDACvtTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_ENUMS(ColorSpace)
    Q_PROPERTY(ColorSpace colorSpace READ get_colorSpace WRITE set_colorSpace RESET reset_colorSpace STORED false)
    Q_PROPERTY(int channel READ get_channel WRITE set_channel RESET reset_channel STORED false)

public:
    enum ColorSpace { Gray = CV_BGR2GRAY,
                      RGBGray = CV_RGB2GRAY,
                      HLS = CV_BGR2HLS,
                      HSV = CV_BGR2HSV,
                      Lab = CV_BGR2Lab,
                      Luv = CV_BGR2Luv,
                      RGB = CV_BGR2RGB,
                      XYZ = CV_BGR2XYZ,
                      YCrCb = CV_BGR2YCrCb,
                      Color = CV_GRAY2BGR };

private:
    BR_PROPERTY(ColorSpace, colorSpace, Gray)
    BR_PROPERTY(int, channel, -1)

    void project(const Template &src, Template &dst) const
    {
        void* const* srcDataPtr = src.m().ptr<void*>();
        int rows = *((int*) srcDataPtr[1]);
        int cols = *((int*) srcDataPtr[2]);
        int type = *((int*) srcDataPtr[3]);
        std::cout << "CVT" << std::endl;
        std::cout << "rows: " << rows << std::endl;
        std::cout << "cols: " << cols << std::endl;

        Mat dstMat = Mat(src.m().rows, src.m().cols, CV_8UC1);
        void** dstDataPtr = dstMat.ptr<void*>();
        dstDataPtr[1] = srcDataPtr[1];
        dstDataPtr[2] = srcDataPtr[2];
        dstDataPtr[3] = srcDataPtr[3];
        *((int*)dstDataPtr[3]) = CV_8UC1; // not sure if the type of the new mat is the same
       
        br::cuda::cudacvt_wrapper(srcDataPtr[0], &dstDataPtr[0], rows, cols);
        dst = dstMat;

        /*
        if (src.m().channels() > 1 || colorSpace == Color) cvtColor(src, dst, colorSpace);
        else dst = src;

        if (channel != -1) {
            std::vector<Mat> mv;
            split(dst, mv);
            dst = mv[channel % (int)mv.size()];
        } */
    }
};

BR_REGISTER(Transform, CUDACvtTransform)

} // namespace br

#include "imgproc/cudacvt.moc"
