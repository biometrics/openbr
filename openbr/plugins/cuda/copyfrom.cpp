#include <iostream>

#include <opencv2/opencv.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace std;

using namespace cv;

// extern CUDA declaration
namespace br { namespace cuda { namespace cudacopyfrom {
  template <typename T> void wrapper(void* src, T* out, int rows, int cols);
}}}

namespace br
{
  class CUDACopyFrom : public UntrainableTransform
  {
    Q_OBJECT

private:
    void project(const Template &src, Template &dst) const
    {
      // pull the data back out of the Mat
      void* const* dataPtr = src.m().ptr<void*>();
      int rows = *((int*)dataPtr[1]);
      int cols = *((int*)dataPtr[2]);
      int type = *((int*)dataPtr[3]);

      Mat dstMat = Mat(rows, cols, type);
      switch(type) {
      case CV_32FC1:
        br::cuda::cudacopyfrom::wrapper(dataPtr[0], dstMat.ptr<float>(), rows, cols);
        break;
      case CV_8UC1:
        br::cuda::cudacopyfrom::wrapper(dataPtr[0], dstMat.ptr<unsigned char>(), rows, cols);
        break;
      case CV_8UC3:
        br::cuda::cudacopyfrom::wrapper(dataPtr[0], dstMat.ptr<unsigned char>(), rows, cols * 3);
        break;
      default:
        cout << "ERR: Invalid image format" << endl;
        break;
      }
      dst = dstMat;
    }
  };

  BR_REGISTER(Transform, CUDACopyFrom);
}

#include "cuda/copyfrom.moc"
