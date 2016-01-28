#include <pthread.h>
#include <semaphore.h>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace cv;
using namespace cv::gpu;

namespace br { namespace cuda {
  class GpuMatManager {
  private:
    int _numMats;
    GpuMat** _mats;         // holds all the mats
    bool** _matTaken;       // holds whether or not they are taken

    pthread_mutex_t* _matTakenLock;            // lock for matTaken table
    pthread_mutex_t* _openCvOperationLock;     // lock for OpenCV upload/download/realloc operations
    sem_t* _matSemaphore;

  public:
    GpuMatManager(int num);

    GpuMat* reserve();
    void upload(GpuMat* reservedMat, Mat& mat);
    void matchDimensions(GpuMat* srcMat, GpuMat* dstMat);
    void download(GpuMat* reservedMat, Mat& dstMat);
    void release(GpuMat* mat);

    ~GpuMatManager();
  };
}}
