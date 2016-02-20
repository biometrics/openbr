#include <iostream>
using namespace std;
#include <pthread.h>

#define CUDA_SAFE_FREE(cudaPtr, errPtr) \
  /*cout << pthread_self() << ": CUDA Free: " << cudaPtr << endl;*/ \
  *errPtr = cudaFree(cudaPtr); \
  if (*errPtr != cudaSuccess) { \
    cout << pthread_self() << ": CUDA Free Error(" << *errPtr << "): " << cudaGetErrorString(*errPtr) << endl; \
    throw 0; \
  }

#define CUDA_SAFE_MALLOC(cudaPtrPtr, size, errPtr) \
  *errPtr = cudaMalloc(cudaPtrPtr, size); \
  if (*errPtr != cudaSuccess) { \
    cout << pthread_self() << ": CUDA Malloc Error(" << *errPtr  << "): " << cudaGetErrorString(*errPtr) << endl; \
    throw 0; \
  } \
  //cout << pthread_self() << ": CUDA Malloc: " << (void*)*(int**)cudaPtrPtr << endl;

#define CUDA_SAFE_MEMCPY(dstPtr, srcPtr, count, kind, errPtr) \
  *errPtr = cudaMemcpy(dstPtr, srcPtr, count, kind); \
  if (*errPtr != cudaSuccess) { \
    cout << pthread_self() << ": CUDA Memcpy Error(" << *errPtr << "): " << cudaGetErrorString(*errPtr) << endl; \
    throw 0; \
  }

#define CUDA_KERNEL_ERR_CHK(errPtr) \
  *errPtr = cudaPeekAtLastError(); \
  if (*errPtr != cudaSuccess) { \
    cout << pthread_self() << ": Kernel Call Err(" << *errPtr << "): " << cudaGetErrorString(*errPtr) << endl; \
    throw 0; \
  }
