namespace br { namespace cuda { namespace cudacopyto {
  //template<typename T>
  //void wrapper(const T* in, void** out, const int rows, const int cols) {
  void wrapper(const unsigned char* in, void** out, const int rows, const int cols) {
    cudaMalloc(out, rows*cols*sizeof(unsigned char));
    cudaMemcpy(*out, in, rows*cols*sizeof(unsigned char), cudaMemcpyHostToDevice);
  }
}}}
