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
using namespace std;

#include <openbr/plugins/openbr_internal.h>

// definitions from the CUDA source file
namespace br { namespace cuda { namespace L2 {
  void wrapper(float const* aPtr, float const* bPtr, int length, float* outPtr);
}}}

namespace br
{

/*!
 * \ingroup distances
 * \brief L2 distance computed using eigen.
 * \author Colin Heinzmann \cite DepthDeluxe
 */
class CUDAL2Distance : public UntrainableDistance
{
    Q_OBJECT

    float compare(const cv::Mat &a, const cv::Mat &b) const
    {
      if (a.type() != CV_32FC1 || b.type() != CV_32FC1) {
        cout << "ERR: Type mismatch" << endl;
        throw 0;
      }
      if (a.rows*a.cols != b.rows*b.cols) {
        cout << "ERR: Dimension mismatch" << endl;
        throw 1;
      }

      float out;
      cuda::L2::wrapper(a.ptr<float>(), b.ptr<float>(), a.rows*a.cols, &out);

      return out;
    }
};

BR_REGISTER(Distance, CUDAL2Distance)

} // namespace br

#include "cuda/cudal2.moc"
