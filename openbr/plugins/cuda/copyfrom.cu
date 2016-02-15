#include "cudadefines.hpp"

namespace br { namespace cuda { namespace cudacopyfrom {
  template <typename T> void wrapper(void* src, T* dst, int rows, int cols) {
    cudaError_t err;
    CUDA_SAFE_MEMCPY(dst, src, rows*cols*sizeof(T), cudaMemcpyDeviceToHost, &err);
    CUDA_SAFE_FREE(src, &err);
  }

  template void wrapper(void*, float*, int, int);
  template void wrapper(void*, unsigned char*, int, int);
}}}
