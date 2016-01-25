#include <openbr/plugins/openbr_internal.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace cv;
using namespace cv::gpu;

#include "passthrough.hpp"

#include <iostream>

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

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
    }
  };

  BR_REGISTER(Transform, CUDAPassthroughTransform);
}

#include "cuda/passthrough.moc"
