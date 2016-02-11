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
    _matTaken = (bool**)malloc(num * sizeof(bool*));
    _matsDimension = (int**)malloc(num * sizeof(int*));

    for (int i=0; i < num; i++) {
      cudaMalloc(&_mats[i], 1 * sizeof(uint8_t));
      //_mats[i] = new GpuMat();

      // initialize matTaken
      _matTaken[i] = new bool;
      (*_matTaken[i]) = false;

      // initialize all mat dimensions to be 1
      _matsDimension[i] = new int;
      (*_matsDimension[i]) = 1;
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

  int MatManager::reserve(Mat *mat) {
    int reservedMatIndex = 0;
    //std::cout << "Reserving" << std::endl << std::flush;

    sem_wait(_matSemaphore);
    pthread_mutex_lock(_matTakenLock);
    int i;
    for (i=0; i < _numMats; i++) {
      if ( !(*_matTaken[i]) ) {
        *_matTaken[i] = true;
        reservedMatIndex = i;
        //std::cout << "Taking " << i << std::endl << std::flush;
        break;
      }
    }
    if (i == _numMats) {
      std::cout << "Cannot reserve a mat. Not enough GpuMat resourses\n" << std::endl << std::flush;
    }

    //printMats();
    //printSemValue();
    pthread_mutex_unlock(_matTakenLock);

    // reallocate if size does not match
    pthread_mutex_lock(_matsDimensionLock);
    if (*_matsDimension[reservedMatIndex] != mat->rows * mat->cols) {
      //printSizeChangingMat(reservedMat);
      //reservedMat->release();
      //reservedMat->create(mat->size(), mat->type());
      //std::cout << "Size mismatch" << std::endl << std::flush;
      // re malloc
      cudaFree(_mats[reservedMatIndex]); // free the previous memory first
      cudaMalloc(&_mats[reservedMatIndex], mat->rows * mat->cols * sizeof(uint8_t));
      // change the dimension of that matrix
      *_matsDimension[reservedMatIndex] = mat->rows * mat->cols;

    }
    pthread_mutex_unlock(_matsDimensionLock);
    return reservedMatIndex;
  }

  void MatManager::upload(int reservedMatIndex, Mat& mat) {
    // upload the image
    /*
    pthread_mutex_lock(_matsDimensionLock);
    reservedMat->upload(mat);
    pthread_mutex_unlock(_matsDimensionLock);
    */

    // copy the content of the Mat to GPU
    uint8_t* reservedMat = _mats[reservedMatIndex];
    cudaMemcpy(reservedMat, mat.ptr<uint8_t>(), mat.rows * mat.cols, cudaMemcpyHostToDevice);
  }

  void MatManager::download(int reservedMatIndex, Mat& dstMat) {
    /*
    pthread_mutex_lock(_matsDimensionLock);
    reservedMat->download(dstMat);
    pthread_mutex_unlock(_matsDimensionLock);
    */

    // copy the mat data back
    int dimension = dstMat.rows * dstMat.cols;
    uint8_t* reservedMat = _mats[reservedMatIndex];
    cudaMemcpy(dstMat.ptr<uint8_t>(), reservedMat, dimension, cudaMemcpyDeviceToHost);
  }

  void MatManager::release(int reservedMatIndex) {
    uint8_t* reservedMat = _mats[reservedMatIndex];
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
      std::cout << "Reservedmat is not in the _mats array" << std::endl << std::flush;
      return;
    }
    /*
    printReleasingMat(reservedMat);
    pthread_mutex_lock(_matsDimensionLock);
    Size size = reservedMat->size();
    int type = reservedMat->type();
    reservedMat->release();
    reservedMat->create(size, type);



    pthread_mutex_unlock(_matsDimensionLock);
    */

    sem_post(_matSemaphore);
  }

  MatManager::~MatManager() {
    // assume a single thread is destroying the manager
    // TODO(colin): add the destroy code
    //std::cout << "Start to destroy.." << std::endl << std::flush;
  }

  /*
  void MatManager::printMats() {
    for (int i = 0; i < _numMats; i++) {
      if ((*_matTaken[i]) == true) {
        std::cout << i << ": Taken, " << _mats[i]->size() << std::endl << std::flush;
      } else {
        std::cout << i << ": Not taken, " << _mats[i]->size() << std::endl << std::flush;
      }
    }
    std::cout << std::endl << std::flush;
  }

  void MatManager::printSemValue() {
    int semValue;
    sem_getvalue(_matSemaphore, &semValue);
    std::cout << "Sem value: " << semValue << std::endl << std::flush;
  }

  void MatManager::printSizeChangingMat(GpuMat* gpuMat) {
    for (int i=0; i < _numMats; i++) {
      if (gpuMat == _mats[i]) {
        std::cout << "changing is size of" << i << " at " << gpuMat <<  std::endl << std::flush;
        return;
      }
    }
    std::cout << "can't change size of mat at address: " << gpuMat << std::endl << std::flush;
  }

  void MatManager::printReleasingMat(GpuMat* gpuMat) {
    for (int i=0; i < _numMats; i++) {
      if (gpuMat == _mats[i]) {
        std::cout << "releasing mat" << i << " at " << gpuMat <<  std::endl << std::flush;
        return;
      }
    }
    std::cout << "can't release mat at address: " << gpuMat << std::endl << std::flush;
  }
*/
  uint8_t* MatManager::get_mat_pointer_from_index(int matIndex) {
    return _mats[matIndex];
  }

}}
