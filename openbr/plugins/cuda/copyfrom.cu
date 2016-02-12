namespace br { namespace cuda { namespace cudacopyfrom {
  //template <typename T> void wrapper(void* src, T* out, int rows, int cols) {
  void wrapper(void* src, unsigned char* out, const int rows, const int cols) {
    cudaMemcpy(out, src, rows*cols*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(src);
  }
}}}
