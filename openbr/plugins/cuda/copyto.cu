/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2016 Colin Heinzmann                                            *
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

#include "cudadefines.hpp"

namespace br { namespace cuda { namespace copyto {

  template <typename T> void wrapper(const T* in, void** out, const int rows, const int cols) {
    cudaError_t err;
    CUDA_SAFE_MALLOC(out, rows*cols*sizeof(T), &err);
    CUDA_SAFE_MEMCPY(*out, in, rows*cols*sizeof(T), cudaMemcpyHostToDevice, &err);
  }

  template void wrapper(const float* in, void** out, const int rows, const int cols);
  template void wrapper(const unsigned char* in, void** out, const int rows, const int cols);

}}}
