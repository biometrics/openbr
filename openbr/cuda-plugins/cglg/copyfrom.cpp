#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace std;

using namespace cv;
using namespace cv::gpu;


namespace br
{
  class CUDACopyFrom : public UntrainableTransform
  {
    Q_OBJECT

private:
    void project(const Template &src, Template &dst) const
    {
      // reassemble the integer and then build pointer to it
      uint64_t gpuMatInt = (((uint64_t)src.m().at<int>(1,0)) << (uint64_t)32) + ((uint64_t)src.m().at<int>(0,0));
      GpuMat* gpuMat = (GpuMat*)gpuMatInt;

      printf("gpuMatInt: %li\n", gpuMatInt);
      printf("m.at(0,0): %i\nm.at(1,0): %i\n", src.m().at<int>(0,0), src.m().at<int>(1,0));

      // download the data back into the destination
      Size size = gpuMat->size();
      Mat out = Mat(size.height, size.width, gpuMat->depth());

      gpuMat->download(out);

      dst = out;
    }
  };

  BR_REGISTER(Transform, CUDACopyFrom);
}

#include "cuda/copyfrom.moc"
