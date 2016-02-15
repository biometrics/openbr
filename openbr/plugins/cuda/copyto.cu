#include "cudadefines.hpp"

namespace br { namespace cuda { namespace cudacopyto {
  template <typename T> void wrapper(const T* in, void** out, const int rows, const int cols) {
    cudaError_t err;
    CUDA_SAFE_MALLOC(out, rows*cols*sizeof(T), &err);
    CUDA_SAFE_MEMCPY(*out, in, rows*cols*sizeof(T), cudaMemcpyHostToDevice, &err);
  }

  template void wrapper(const float* in, void** out, const int rows, const int cols);
  template void wrapper(const unsigned char* in, void** out, const int rows, const int cols);
}}}
