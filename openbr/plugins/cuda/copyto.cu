namespace br { namespace cuda { namespace cudacopyto {
  template <typename T> void wrapper(const T* in, void** out, const int rows, const int cols) {
    cudaMalloc(out, rows*cols*sizeof(T));
    cudaMemcpy(*out, in, rows*cols*sizeof(T), cudaMemcpyHostToDevice);
  }

  template void wrapper(const float* in, void** out, const int rows, const int cols);
  template void wrapper(const unsigned char* in, void** out, const int rows, const int cols);
}}}
