#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace std;

using namespace cv;
using namespace cv::gpu;

namespace br
{
  class CUDAThreshold : public UntrainableTransform
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

  BR_REGISTER(Transform, CUDAThreshold);
}

#include "cuda/threshold.moc"
