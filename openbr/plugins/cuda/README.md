# CUDA Plugins
CUDA plugins are very similar to normal plugins.  A single plugin is split into
two files: the `.cpp` file with the BR standard plugin definition and the `.cu`
file with your kernel and wrapper functions.

## The `.cpp` file
Every main plugin file must have the names of the kernel wrapper functions
defined at the top of the program.  Once the definitions are there, just call
the CUDA functions as you need them

## The `.cu` file
All functions within the CUDA file must be declared inside their own namespace
under `br::cuda`.  For example the plugin `passthrough` must have all functions
inside it declared under the namespace `br::cuda::passthrough`.

## CPU Template object format
Like any other BR Transform, the plugin must return an object for the next
plugin to consume.  For performance reasons, we don't copy data to and from
the graphics card for every transform.  Instead, we use this space to transfer
data about how to access the image data and its type.  The Mat is an array of data type `void*`.

Index   | Item Name   | Type      | Description
--------|-------------|-----------|------------
0       | GpuData     | void*     | Pointer to the graphics card data
1       | rows        | int       | Number of rows in the Mat
2       | cols        | int       | Number of colums in the Mat
3       | type        | int       | OpenCV mat data type code (i.e. `mat.type()`)

It is expected that the wrapper function does the proper GPU memory handling
to make sure that the GpuData pointer in the output mat is pointing to the
data that the plugin is outputting.

## Example: Passthrough
This example plugin takes in input data and passes it straight to the output.
The BR transform calls the wrapper function which exists in the CUDA file which
in turn calls the kernel routine to copy the data in the GPU.

**Note**: This program assumes that a previous Transform, namely `CUDACopyTo` has
copied the data to the GPU.

### **passthrough.cpp**
```c++
#include <openbr/plugins/openbr_internal.h>
#include <opencv2/opencv.hpp>

// wrapper function within the CUDA file
namespace br { namespace cuda { namespace passthrough {
  void wrapper(void* srcGpuData, void** dstGpuData);
}}};

#include <iostream>
namespace br
{
  class CUDAPassthroughTransform : public UntrainableTransform
  {
    Q_OBJECT

    void project(const Template &src, Template &dst) {
      // extract the parameters out of the Mat passed from the previous plugin
      void* const* srcDataPtr = src.m().ptr<void*>();
      int rows = *((int*)srcDataPtr[1]);
      int cols = *((int*)srcDataPtr[2]);
      int type = *((int*)srcDataPtr[3]);

      // generate a new Mat to be passed to the next plugin
      Mat dstMat = Mat(src.m().rows, src.m().cols, src.m().type());
      void** dstDataPtr = dstMat.ptr<void*>();
      dstDataPtr[1] = srcDataPtr[1];
      dstDataPtr[2] = srcDataPtr[2];
      dstDataPtr[3] = srcDataPtr[3];

      // call the wrapper and set the dst output to the newly created Mat
      br::cuda::passthrough::wrapper(srcDataPtr[0], &dstDataPtr[0], rows, cols);
      dst = dstMat;
    }
  };

  BR_REGISTER(Transform, CUDAPassthroughTransform);
}

#include "cuda/passthrough.moc"
```

### **passthrough.cu**
```c++
#include <opencv2/opencv.hpp>

namespace br { namespace cuda { namespace passthrough {
  __global__ void kernel(char* srcPtr, char* dstPtr, int rows, int cols) {
    // get the current index
    int rowInd = blockIdx.y*blockDim.y+threadIdx.y;
    int colInd = blockIdx.x*blockDim.x+threadIdx.x;

    // don't do anything if we are outside the allowable positions
    if (rowInd >= rows || colInd >= cols)
      return;

    // write the input to the output
    rowDstPtr[rowInd*cols + colInd] = srcVal;
  }

  void wrapper(char* srcPtr, char** dstPtr, int rows, int cols, int type) {
    // verify the proper image type
    if (type != CV_8UC1) {
      cout << "Error: image type not supported"
      return;
    }

    *dstPtr = cudaMalloc(rows*cols*sizeof(char));

    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks(imageWidth / threadsPerBlock.x + 1,
                   imageHeight / threadsPerBlock.y + 1);

    // run the kernel function
    kernel<<<numBlocks, threadPerBlock>>>(srcPtr, dstPtr, rows, cols);

    // free the memory as it isn't used anymore
    cudaFree(srcPtr);
  }
}}}
```
