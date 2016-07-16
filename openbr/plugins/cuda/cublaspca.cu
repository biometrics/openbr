namespace br { namespace cuda { namespace pca {
  __global__ void castFloatToDoubleKernel(float* a, int inca, double* b, int incb, int numElems) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    if (index >= numElems) {
      return;
    }

    b[index*incb] = (double)a[index*inca];
  }

  __global__ void castDoubleToFloatKernel(double* a, int inca, float* b, int incb, int numElems) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    if (index >= numElems) {
      return;
    }

    b[index*incb] = (float)a[index*inca];
  }

  void castFloatToDouble(float* a, int inca, double* b, int incb, int numElems) {
    int threadsPerBlock = 512;
    int numBlocks = numElems / threadsPerBlock + 1;

    castFloatToDoubleKernel<<<numBlocks, threadsPerBlock>>>(a, inca, b, incb, numElems);
  }

  void castDoubleToFloat(double* a, int inca, float* b, int incb, int numElems) {
    int threadsPerBlock = 512;
    int numBlocks = numElems / threadsPerBlock + 1;

    castDoubleToFloatKernel<<<numBlocks, threadsPerBlock>>>(a, inca, b, incb, numElems);
  }
}}}
