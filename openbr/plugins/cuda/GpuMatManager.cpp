#include <pthread.h>
#include <semaphore.h>

#include <opencv2/opencv.hpp>

#include "GpuMatManager.hpp"

using namespace cv;
using namespace cv::gpu;

namespace br { namespace cuda {
  GpuMatManager::GpuMatManager(int num) {
    _numMats = num;

    // initialize the GpuMats
    _mats = (GpuMat**)malloc(num * sizeof(GpuMat*));
    _matTaken = (bool**)malloc(num * sizeof(bool*));
    for (int i=0; i < num; i++) {
      _mats[i] = new GpuMat();
      _matTaken[i] = new bool;
      (*_matTaken[i]) = false;
    }

    // initialize the locks
    _matTakenLock = new pthread_mutex_t;
    pthread_mutex_init(_matTakenLock, NULL);
    _openCvOperationLock = new pthread_mutex_t;
    pthread_mutex_init(_openCvOperationLock, NULL);

    // initialize the semaphore
    _matSemaphore = new sem_t;
    sem_init(_matSemaphore, 0, _numMats);
  }

  GpuMat* GpuMatManager::reserve() {
    GpuMat* reservedMat = NULL;

    // get the reserved GpuMat
    //sem_wait(_matSemaphore);
    pthread_mutex_lock(_matTakenLock);
    for (int i=0; i < _numMats; i++) {
      if ( !(*_matTaken[i]) ) {
        reservedMat = _mats[i];
        *_matTaken[i] = true;
        break;
      }
    }
    pthread_mutex_unlock(_matTakenLock);

    return reservedMat;
  }

  void GpuMatManager::upload(GpuMat* reservedMat, Mat& mat) {
    // check the image Dimensions
    if (reservedMat->size() != mat.size()) {
      pthread_mutex_lock(_openCvOperationLock);
      reservedMat->release();
      reservedMat->create(mat.size(), mat.type());
      pthread_mutex_unlock(_openCvOperationLock);
    }

    // upload the image
    pthread_mutex_lock(_openCvOperationLock);
    reservedMat->upload(mat);
    pthread_mutex_unlock(_openCvOperationLock);
    pthread_mutex_lock(_openCvOperationLock);
    reservedMat->upload(mat);
    pthread_mutex_unlock(_openCvOperationLock);
  }

  void GpuMatManager::matchDimensions(GpuMat* srcMat, GpuMat* dstMat) {
    if (srcMat->size() != dstMat->size()) {
      pthread_mutex_lock(_openCvOperationLock);
      dstMat->release();
      dstMat->create(srcMat->size(), srcMat->type());
      pthread_mutex_unlock(_openCvOperationLock);
    }
  }

  void GpuMatManager::download(GpuMat* reservedMat, Mat& dstMat) {
    pthread_mutex_lock(_openCvOperationLock);
    reservedMat->download(dstMat);
    pthread_mutex_unlock(_openCvOperationLock);
  }

  void GpuMatManager::release(GpuMat* reservedMat) {
    pthread_mutex_lock(_matTakenLock);
    bool foundMatch = false;
    for (int i=0; i < _numMats; i++) {
      if (reservedMat == _mats[i]) {
        *_matTaken[i] = false;
        foundMatch = true;
      }
    }
    pthread_mutex_unlock(_matTakenLock);

    // return unconditionally if we didn't find a match
    if (!foundMatch) {
      return;
    }

    sem_post(_matSemaphore);
  }

  GpuMatManager::~GpuMatManager() {
    // assume a single thread is destroying the manager
    // TODO(colin): add the destroy code
  }

}}
