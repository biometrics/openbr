#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace std;

using namespace cv;
using namespace cv::gpu;

namespace br
{
  class CUDACopyTo : public UntrainableTransform
  {
    Q_OBJECT

private:
    void project(const Template &src, Template &dst) const
    {
      // get the mat to send to the GPU
      GpuMat* gpuMat = new GpuMat;

      try
      {
        // copy the contents to the GPU
        gpuMat->upload(src.m());
      }
      catch(const cv::Exception& ex)
      {
        cout << "Error: " << ex.what() << endl;
      }

      // now create a new Mat that contains the 64-bit pointer
      Mat m = Mat(2, 1, CV_32S);

      // pointer magic
      uint64_t gpuMatInt = (uint64_t)gpuMat;
      m.at<int>(0,0) = (int32_t)(gpuMatInt &  0x00000000FFFFFFFF);
      m.at<int>(1,0) = (int32_t)((gpuMatInt & 0xFFFFFFFF00000000) >> (uint64_t)32);

      printf("gpuMatInt: %li\n", gpuMatInt);
      printf("m.at(0,0): %i\nm.at(1,0): %i\n", m.at<int>(0,0), m.at<int>(1,0));

      // save away in the destination mat
      dst += m;
    }
  };

  BR_REGISTER(Transform, CUDACopyTo);
}

#include "cuda/copyto.moc"
