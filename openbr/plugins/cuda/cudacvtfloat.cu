namespace br { namespace cuda { namespace cudacvtfloat {

  __global__ void kernel(const unsigned char* src, float* dst, int rows, int cols) {
    // get my index
    int rowInd = blockIdx.y*blockDim.y + threadIdx.y;
    int colInd = blockIdx.x*blockDim.x + threadIdx.x;

    // bounds check
    if (rowInd >= rows || colInd >= cols) {
      return;
    }

    int index = rowInd*cols + colInd;
    dst[index] = (float)src[index];
  }

  void wrapper(const unsigned char* src, void** dst, int rows, int cols) {
    //unsigned char* cudaSrc;
    //cudaMalloc(&cudaSrc, rows*cols*sizeof(unsigned char));
    //cudaMemcpy(cudaSrc, src, rows*cols*sizeof(unsigned char), cudaMemcpyHostToDevice);

    //float* cudaDst;
    //cudaMalloc(&cudaDst, rows*cols*sizeof(float));

    cudaMalloc(dst, rows*cols*sizeof(float));

    dim3 threadsPerBlock(8, 8);
    dim3 blocks(
      cols / threadsPerBlock.x + 1,
      rows / threadsPerBlock.y + 1
    );

    kernel<<<threadsPerBlock, blocks>>>(src, (float*)(*dst), rows, cols);
  }

}}}
