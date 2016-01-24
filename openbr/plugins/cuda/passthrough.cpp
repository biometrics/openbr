#include <openbr/plugins/openbr_internal.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace cv;
using namespace cv::gpu;

extern void br_cuda_device_wrapper();

namespace br
{
  class CUDAPassthroughTransform : public UntrainableTransform
  {
    Q_OBJECT

private:
    void project(const Template &src, Template &dst) const
    {
      // upload the src mat to the GPU
      GpuMat srcGpuMat, dstGpuMat;
      srcGpuMat.upload(src.m());
      dstGpuMat.upload(src.m());

      br_cuda_device_wrapper();

      dstGpuMat.download(dst.m());

      // TODO(colin): add delete code
    }
  };

  BR_REGISTER(Transform, CUDAPassthroughTransform);
}

#include "cuda/passthrough.moc"
