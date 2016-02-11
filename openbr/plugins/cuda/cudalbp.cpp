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
//#include <thread>
//#include <mutex>

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

#include "cudalbp.hpp"
#include "MatManager.hpp"

using namespace cv;

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

int ctr = 0;
pthread_mutex_t* uploadMutex = NULL;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Convert the image into a feature vector using Local Binary Patterns
 * \br_paper Ahonen, T.; Hadid, A.; Pietikainen, M.;
 *           "Face Description with Local Binary Patterns: Application to Face Recognition"
 *           Pattern Analysis and Machine Intelligence, IEEE Transactions, vol.28, no.12, pp.2037-2041, Dec. 2006
 * \author Josh Klontz \cite jklontz
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
    uint8_t* lutGpuPtr;
    uchar null;


    cuda::MatManager* matManager;

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

        // init the mat manager for managing 10 mats
        matManager = new cuda::MatManager(10);

        // copy lut over to the GPU
        br::cuda::cudalbp_init_wrapper(lut, &lutGpuPtr);

        std::cout << "Initialized CUDALBP" << std::endl;
    }

    void project(const Template &src, Template &dst) const
    {
        Mat& m = (Mat&)src.m();
        cuda::MatManager::matindex a;
        cuda::MatManager::matindex b;
        a = matManager->reserve(m);
        matManager->upload(a, m);

        // reserve the second mat and check the dimensiosn
        b = matManager->reserve(m);
        
        uint8_t* srcMatPtr = matManager->get_mat_pointer_from_index(a);
        uint8_t* dstMatPtr = matManager->get_mat_pointer_from_index(b);
        br::cuda::cudalbp_wrapper(srcMatPtr, dstMatPtr, lutGpuPtr, m.cols, m.rows, m.step1());

        matManager->download(b, dst);

        // release both the mats
        matManager->release(a);
        matManager->release(b);
    }
};

BR_REGISTER(Transform, CUDALBPTransform)

} // namespace br

#include "cuda/cudalbp.moc"
