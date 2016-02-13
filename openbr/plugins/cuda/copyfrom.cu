namespace br { namespace cuda { namespace cudacopyfrom {
  //template <typename T> void wrapper(void* src, T* out, int rows, int cols) {
  void wrapper(void* src, float* dst, const int rows, const int cols) {
    cudaMemcpy(dst, src, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(src);
  }
}}}
