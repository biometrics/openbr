/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2016 Colin Heinzmann                                     *
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
