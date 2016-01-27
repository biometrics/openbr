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

    //std::mutex uploadMutex;
    pthread_mutex_t* uploadMutex;

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
        br::cuda::cudalbp_init_wrapper(lut, &lutGpuPtr);

        // initialize the mutex
        std::cout << "STARING EVERYTHING" << std::endl<< std::flush;
        if (uploadMutex == NULL) {
          uploadMutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
          pthread_mutex_init(uploadMutex, NULL);
        }
    }

    void project(const Template &src, Template &dst) const
    {
        int myCtr = ctr++;
        GpuMat a, b;
        const Mat& m = src.m();

        std::cout << "PID: " << getpid() << std::endl << std::flush;

        //std::cout << "START: " << myCtr << std::endl << std::flush;


        //std::cout << "Image type: " << type2str(m.type()) << std::endl << std::flush;
        pthread_mutex_lock(uploadMutex);
        a.create(m.size(), m.type());
        b.create(m.size(), m.type());
        pthread_mutex_unlock(uploadMutex);

        pthread_mutex_lock(uploadMutex);
        a.upload(m);
        b.upload(m);
        pthread_mutex_unlock(uploadMutex);

        // resize the mats
        //if (m.size() != srcGpuMat->size()) {
        //  printf("resizing...\n");
        //  srcGpuMat->release();                    dstGpuMat->release();
        //  srcGpuMat->create(m.size(), CV_8UC1);    dstGpuMat->create(m.size(), CV_8UC1);
        //}

        // copy the data to the GPU
        //srcGpuMat->upload(m);

        // call the kernel function
        //br::cuda::cudalbp_wrapper(*srcGpuMat, *dstGpuMat, lutGpuPtr);
        pthread_mutex_lock(uploadMutex);
        br::cuda::cudalbp_wrapper(a, b, lutGpuPtr);
        pthread_mutex_unlock(uploadMutex);

        // download the result to the destination
        //dstGpuMat->download(dst.m());
        pthread_mutex_lock(uploadMutex);
        b.download(dst.m());
        pthread_mutex_unlock(uploadMutex);

        pthread_mutex_lock(uploadMutex);
        a.release();
        b.release();
        pthread_mutex_unlock(uploadMutex);
    }
};

BR_REGISTER(Transform, CUDALBPTransform)

} // namespace br

#include "cuda/cudalbp.moc"
