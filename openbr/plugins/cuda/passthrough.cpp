#include <openbr/plugins/openbr_internal.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace cv;
using namespace cv::gpu;

#include "passthrough.hpp"

#include <iostream>


namespace br
{
  class CUDAPassthroughTransform : public UntrainableTransform
  {
    Q_OBJECT

private:
    void project(const Template &src, Template &dst) const
    {
      // note: if you convert the image to grayscale, you get 8UC1

      // upload the src mat to the GPU
      GpuMat srcGpuMat, dstGpuMat;
      srcGpuMat.upload(src.m());
      dstGpuMat.upload(src.m());

      br::cuda::passthrough_wrapper(srcGpuMat, dstGpuMat);

      dstGpuMat.download(dst.m());

      // TODO(colin): add delete code
      srcGpuMat.release();
      dstGpuMat.release();

      printf("srcGpuMat empty: %d\n", (int)srcGpuMat.empty());
      printf("dstGpuMat empty: %d\n", (int)srcGpuMat.empty());
    }
  };

  BR_REGISTER(Transform, CUDAPassthroughTransform);
}

#include "cuda/passthrough.moc"
