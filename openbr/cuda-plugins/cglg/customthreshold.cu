/*
#include <iostream>

// external opencv CUDA interface
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

// internal CUDA stuff
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/emulation.hpp>
#include <opencv2/core/cuda/transform.hpp>
#include <opencv2/core/cuda/functional.hpp>
#include <opencv2/core/cuda/utility.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace std;

using namespace cv;
using namespace cv::gpu;
using namespace cv::cuda;
using namespace cv::cuda::device;

namespace br
{
  class CUDACustomThresholdTransform : public UntrainableTransform
  {
    Q_OBJECT

private:
    void project(const Template &src, Template &dst) const
    {
      // get the mat to send to the GPU
      GpuMat gpuMat_src, gpuMat_dst;

      try
      {
        // copy the contents to the GPU
        gpuMat_src.upload(src.m());

        threshold(gpuMat_src, gpuMat_dst, 128.0, 255.0, CV_THRESH_BINARY);

        gpuMat_dst.download(dst.m());
      }
      catch(const cv::Exception& ex)
      {
        cout << "Error: " << ex.what() << endl;
      }
    }
  };

  BR_REGISTER(Transform, CUDACustomThresholdTransform);

  namespace cuda { namespace customthreshold {
    texture<uchar, cudaTextureType2D, cudaReadModeElementType> tex_src(false, cudaFilterModePoint, cudaAddressModeClamp);
    struct SrcTex {
      __host__ SrcTex(int _xoff, int _yoff) : xoff(_xoff), yoff(_yoff) {}
      __device__ __forceinline__ int operator ()(int y, int x) const {
        return tex2D(tex_src, x + xoff, y + yoff);
      }
    }
    __global__ void testKernel(const SrcTex src) {
      const int x = blockIdx.x * blockDim.x + threadIdx.x;
      const int y = blockIdx.y * blockDim.y + threadIdx.y;

      src(x, y) = 1;sajfflksajlkfjdsalkfjsadjflkdsaf
    }
  }
}

#include "cglg/customthreshold.moc"
*/
