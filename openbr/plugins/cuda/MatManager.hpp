/*
NOTES
Mat reservations should return a handle instead of a pointer
*/

#include <pthread.h>
#include <semaphore.h>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace cv;
using namespace cv::gpu;

namespace br { namespace cuda {
  class MatManager {
  private:
    int _numMats;
    uint8_t** _mats;         // holds all the mats
    bool** _matTaken;       // holds whether or not they are taken
    int** _matsDimension;   // holds the dimension of the Mats

    pthread_mutex_t* _matTakenLock;            // lock for matTaken table
    pthread_mutex_t* _matsDimensionLock;     // lock for OpenCV upload/download/realloc operations
    sem_t* _matSemaphore;

  public:
    typedef int matindex;
    MatManager(int num);

    int reserve(Mat &mat);
    void upload(matindex reservedMatIndex, Mat& mat);
    void download(matindex reservedMatIndex, Mat& dstMat);
    void release(matindex matIndex);
    uint8_t* get_mat_pointer_from_index(matindex matIndex);

    ~MatManager();
    //void printMats();
    //void printSemValue();
    //void printSizeChangingMat(uint8_t* gpuMat);
    //void printReleasingMat(uint8_t* gpuMat);
  };
}}
