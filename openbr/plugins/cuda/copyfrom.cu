namespace br { namespace cuda { namespace cudacopyfrom {
  template <typename T> void wrapper(void* src, T* dst, int rows, int cols) {
    cudaMemcpy(dst, src, rows*cols*sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(src);
  }

  template void wrapper(void*, float*, int, int);
  template void wrapper(void*, unsigned char*, int, int);
}}}
