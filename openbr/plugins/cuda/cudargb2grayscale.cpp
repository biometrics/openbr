/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2016 Li Li, Colin Heinzmann                                     *
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

// definitions from the CUDA source file
namespace br { namespace cuda { namespace rgb2grayscale {
  void wrapper(void* srcPtr, void**dstPtr, int rows, int cols);
}}}

namespace br
{

/*!
 * \ingroup transforms
 * \brief Converts 3-channel images to grayscale
 * \author Li Li \cite booli
 */
class CUDARGB2GrayScaleTransform : public UntrainableTransform
{
    Q_OBJECT

public:

private:
    void project(const Template &src, Template &dst) const
    {
        void* const* srcDataPtr = src.m().ptr<void*>();
        int rows = *((int*) srcDataPtr[1]);
        int cols = *((int*) srcDataPtr[2]);
        int type = *((int*) srcDataPtr[3]);

        Mat dstMat = Mat(src.m().rows, src.m().cols, src.m().type());
        void** dstDataPtr = dstMat.ptr<void*>();
        dstDataPtr[1] = srcDataPtr[1];
        dstDataPtr[2] = srcDataPtr[2];
        dstDataPtr[3] = srcDataPtr[3];
        *((int*)dstDataPtr[3]) = CV_8UC1; // not sure if the type of the new mat is the same

        cuda::rgb2grayscale::wrapper(srcDataPtr[0], &dstDataPtr[0], rows, cols);
        dst = dstMat;
    }
};

BR_REGISTER(Transform, CUDARGB2GrayScaleTransform)

} // namespace br

#include "imgproc/cudargb2grayscale.moc"
