/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2016 Colin Heinzmann                                            *
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

#include <opencv2/opencv.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace std;

using namespace cv;

// definitions from the CUDA source file
namespace br { namespace cuda { namespace copyto {
  template <typename T> void wrapper(const T* in, void** out, const int rows, const int cols);
}}}

namespace br
{

  /*!
  * \ingroup transforms
  * \brief Copies a transform to the GPU.
  * \author Colin Heinzmann \cite DepthDeluxe
  * \note Method: Automatically matches image dimensions, works for 32-bit single channel, 8-bit single channel, and 8-bit 3 channel
  */
  class CUDACopyTo : public UntrainableTransform
  {
    Q_OBJECT

private:
    void project(const Template &src, Template &dst) const
    {
      const Mat& srcMat = src.m();
      const int rows = srcMat.rows;
      const int cols = srcMat.cols;

      // output will be a single pointer to graphics card memory
      Mat dstMat = Mat(4, 1, DataType<void*>::type);
      void** dstMatData = dstMat.ptr<void*>();

      // save cuda ptr, rows, cols, then type
      dstMatData[1] = new int; *((int*)dstMatData[1]) = rows;
      dstMatData[2] = new int; *((int*)dstMatData[2]) = cols;
      dstMatData[3] = new int; *((int*)dstMatData[3]) = srcMat.type();

      void* cudaMemPtr;
      switch(srcMat.type()) {
      case CV_32FC1:
        cuda::copyto::wrapper(srcMat.ptr<float>(), &dstMatData[0], rows, cols);
        break;
      case CV_8UC1:
        cuda::copyto::wrapper(srcMat.ptr<unsigned char>(), &dstMatData[0], rows, cols);
        break;
      case CV_8UC3:
        cuda::copyto::wrapper(srcMat.ptr<unsigned char>(), &dstMatData[0], rows, 3*cols);
        break;
      default:
        cout << "ERR: Invalid image type (" << srcMat.type() << ")" << endl;
        return;
      }

      dst = dstMat;
    }
  };

  BR_REGISTER(Transform, CUDACopyTo);
}

#include "cuda/copyto.moc"
