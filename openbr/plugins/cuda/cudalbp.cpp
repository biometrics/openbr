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
using namespace std;

#include <sys/types.h>
#include <unistd.h>

#include <pthread.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <limits>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

// definitions from the CUDA source file
namespace br { namespace cuda { namespace lbp {
  void wrapper(void* srcPtr, void** dstPtr, int rows, int cols);
  void initializeWrapper(uint8_t* lut);
}}}

namespace br
{
/*!
 * \ingroup transforms
 * \brief Convert the image into a feature vector using Local Binary Patterns in CUDA.  Modified from stock OpenBR plugin.
 * \author Colin Heinzmann \cite DepthDeluxe
 * \author Li Li \cite booli
 */
class CUDALBPTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int radius READ get_radius WRITE set_radius RESET reset_radius STORED false)
    Q_PROPERTY(int maxTransitions READ get_maxTransitions WRITE set_maxTransitions RESET reset_maxTransitions STORED false)
    Q_PROPERTY(bool rotationInvariant READ get_rotationInvariant WRITE set_rotationInvariant RESET reset_rotationInvariant STORED false)
    BR_PROPERTY(int, radius, 1)
    BR_PROPERTY(int, maxTransitions, 8)
    BR_PROPERTY(bool, rotationInvariant, false)

  private:
    uchar lut[256];
    uchar null;

  public:
    /* Returns the number of 0->1 or 1->0 transitions in i */
    static int numTransitions(int i)
    {
        int transitions = 0;
        int curParity = i%2;
        for (int j=1; j<=8; j++) {
            int parity = (i>>(j%8)) % 2;
            if (parity != curParity) transitions++;
            curParity = parity;
        }
        return transitions;
    }

    static int rotationInvariantEquivalent(int i)
    {
        int min = std::numeric_limits<int>::max();
        for (int j=0; j<8; j++) {
            bool parity = i % 2;
            i = i >> 1;
            if (parity) i+=128;
            min = std::min(min, i);
        }
        return min;
    }

    void init()
    {
        bool set[256];
        uchar uid = 0;
        for (int i=0; i<256; i++) {
            if (numTransitions(i) <= maxTransitions) {
                int id;
                if (rotationInvariant) {
                    int rie = rotationInvariantEquivalent(i);
                    if (i == rie) id = uid++;
                    else          id = lut[rie];
                } else            id = uid++;
                lut[i] = id;
                set[i] = true;
            } else {
                set[i] = false;
            }
        }

        null = uid;
        for (int i=0; i<256; i++)
            if (!set[i])
                lut[i] = null; // Set to null id

        // copy lut over to the GPU
        cuda::lbp::initializeWrapper(lut);

        std::cout << "Initialized CUDALBP" << std::endl;
    }

    void project(const Template &src, Template &dst) const
    {
        void* const* srcDataPtr = src.m().ptr<void*>();
        int rows = *((int*)srcDataPtr[1]);
        int cols = *((int*)srcDataPtr[2]);
        int type = *((int*)srcDataPtr[3]);

        Mat dstMat = Mat(src.m().rows, src.m().cols, src.m().type());
        void** dstDataPtr = dstMat.ptr<void*>();
        dstDataPtr[1] = srcDataPtr[1];
        dstDataPtr[2] = srcDataPtr[2];
        dstDataPtr[3] = srcDataPtr[3];

        cuda::lbp::wrapper(srcDataPtr[0], &dstDataPtr[0], rows, cols);
        dst = dstMat;
    }
};

BR_REGISTER(Transform, CUDALBPTransform)

}

#include "cuda/cudalbp.moc"
