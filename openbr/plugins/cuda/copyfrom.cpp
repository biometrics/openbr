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

// CUDA functions for this plugin
namespace br { namespace cuda { namespace copyfrom {
  template <typename T> void wrapper(void* src, T* out, int rows, int cols);
}}}

namespace br
{
  /*!
  * \ingroup transforms
  * \brief Copies a transform from the GPU to the CPU.
  * \author Colin Heinzmann \cite DepthDeluxe
  * \note Method: Automatically matches image dimensions, works for 32-bit single channel, 8-bit single channel, and 8-bit 3 channel
  */
  class CUDACopyFrom : public UntrainableTransform
  {
    Q_OBJECT

private:
    void project(const Template &src, Template &dst) const
    {
      // pull the data back out of the Mat
      void* const* dataPtr = src.m().ptr<void*>();
      int rows = *((int*)dataPtr[1]);
      int cols = *((int*)dataPtr[2]);
      int type = *((int*)dataPtr[3]);

      Mat dstMat = Mat(rows, cols, type);
      switch(type) {
      case CV_32FC1:
        cuda::copyfrom::wrapper(dataPtr[0], dstMat.ptr<float>(), rows, cols);
        break;
      case CV_8UC1:
        cuda::copyfrom::wrapper(dataPtr[0], dstMat.ptr<unsigned char>(), rows, cols);
        break;
      case CV_8UC3:
        cuda::copyfrom::wrapper(dataPtr[0], dstMat.ptr<unsigned char>(), rows, cols * 3);
        break;
      default:
        cout << "ERR: Invalid image type (" << type << ")" << endl;
        break;
      }
      dst = dstMat;
    }
  };

  BR_REGISTER(Transform, CUDACopyFrom);
}

#include "cuda/copyfrom.moc"
