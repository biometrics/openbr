#include <pthread.h>
#include <semaphore.h>

#include <opencv2/opencv.hpp>

#include "MatManager.hpp"

using namespace cv;
using namespace cv::gpu;

namespace br { namespace cuda {
  MatManager::MatManager(int num) {
    _numMats = num;

    // initialize the an array of Mats
    _mats = (uint8_t**)malloc(num * sizeof(uint8_t*));
    _matTaken = (bool*)malloc(num * sizeof(bool));
    _matsDimension = (int*)malloc(num * sizeof(int));

    for (int i=0; i < num; i++) {
      cudaMalloc(&_mats[i], 1 * sizeof(uint8_t));

      // initialize matTaken
      _matTaken[i] = false;

      // initialize all mat dimensions to be 1
      _matsDimension[i] = 1;
    }

    // initialize the locks
    _matTakenLock = new pthread_mutex_t;
    pthread_mutex_init(_matTakenLock, NULL);
    _matsDimensionLock = new pthread_mutex_t;
    pthread_mutex_init(_matsDimensionLock, NULL);

    // initialize the semaphore
    _matSemaphore = new sem_t;
    sem_init(_matSemaphore, 0, _numMats);
  }

  MatManager::matindex MatManager::reserve(Mat &mat) {
    int reservedMatIndex = 0;

    sem_wait(_matSemaphore);
    pthread_mutex_lock(_matTakenLock);
    int i;
    for (i=0; i < _numMats; i++) {
      if ( !_matTaken[i] ) {
        _matTaken[i] = true;
        reservedMatIndex = i;
        break;
      }
    }
    if (i == _numMats) {
      std::cout << "Cannot reserve a mat. Not enough GpuMat resourses\n" << std::endl << std::flush;
    }

    pthread_mutex_unlock(_matTakenLock);

    // reallocate if size does not match
    pthread_mutex_lock(_matsDimensionLock);
    if (_matsDimension[reservedMatIndex] != mat.rows * mat.cols) {
      cudaFree(_mats[reservedMatIndex]); // free the previous memory first
      cudaMalloc(&_mats[reservedMatIndex], mat.rows * mat.cols * sizeof(uint8_t));
      // change the dimension of that matrix
      _matsDimension[reservedMatIndex] = mat.rows * mat.cols;

    }
    pthread_mutex_unlock(_matsDimensionLock);
    return reservedMatIndex;
  }

  void MatManager::upload(MatManager::matindex reservedMatIndex, Mat& mat) {
    // copy the content of the Mat to GPU
    uint8_t* reservedMat = _mats[reservedMatIndex];
    cudaMemcpy(reservedMat, mat.ptr<uint8_t>(), mat.rows * mat.cols, cudaMemcpyHostToDevice);
  }

  void MatManager::download(MatManager::matindex reservedMatIndex, Mat& dstMat) {
    // copy the mat data back
    int dimension = dstMat.rows * dstMat.cols;
    uint8_t* reservedMat = _mats[reservedMatIndex];
    cudaMemcpy(dstMat.ptr<uint8_t>(), reservedMat, dimension, cudaMemcpyDeviceToHost);
  }

  void MatManager::release(MatManager::matindex reservedMatIndex) {
    uint8_t* reservedMat = _mats[reservedMatIndex];
    pthread_mutex_lock(_matTakenLock);
    bool foundMatch = false;
    for (int i=0; i < _numMats; i++) {
      if (reservedMat == _mats[i]) {
        _matTaken[i] = false;
        foundMatch = true;
      }
    }
    pthread_mutex_unlock(_matTakenLock);

    // return unconditionally if we didn't find a match
    if (!foundMatch) {
      std::cout << "Reservedmat is not in the _mats array" << std::endl << std::flush;
      return;
    }
    sem_post(_matSemaphore);
  }

  MatManager::~MatManager() {
    // assume a single thread is destroying the manager
    // TODO(colin): add the destroy code
    //std::cout << "Start to destroy.." << std::endl << std::flush;
  }

  uint8_t* MatManager::get_mat_pointer_from_index(MatManager::matindex matIndex) {
    return _mats[matIndex];
  }

}}
