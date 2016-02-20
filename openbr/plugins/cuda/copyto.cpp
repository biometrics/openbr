#include <iostream>

#include <opencv2/opencv.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace std;

using namespace cv;

extern string type2str(int type);

namespace br { namespace cuda { namespace cudacopyto {
  template <typename T> void wrapper(const T* in, void** out, const int rows, const int cols);
}}}

namespace br
{
  class CUDACopyTo : public UntrainableTransform
  {
    Q_OBJECT

private:
    void project(const Template &src, Template &dst) const
    {
      const Mat& srcMat = src.m();
      const int rows = srcMat.rows;
      const int cols = srcMat.cols; 

      // output will be a single pointer to graphics card memory
      Mat dstMat = Mat(4, 1, DataType<void*>::type);
      void** dstMatData = dstMat.ptr<void*>();

      // save cuda ptr, rows, cols, then type
      dstMatData[1] = new int; *((int*)dstMatData[1]) = rows;
      dstMatData[2] = new int; *((int*)dstMatData[2]) = cols;
      dstMatData[3] = new int; *((int*)dstMatData[3]) = srcMat.type();

      void* cudaMemPtr;
      switch(srcMat.type()) {
      case CV_32FC1:
        br::cuda::cudacopyto::wrapper(srcMat.ptr<float>(), &dstMatData[0], rows, cols);
        break;
      case CV_8UC1:
        br::cuda::cudacopyto::wrapper(srcMat.ptr<unsigned char>(), &dstMatData[0], rows, cols);
        break;
      case CV_8UC3:
        br::cuda::cudacopyto::wrapper(srcMat.ptr<unsigned char>(), &dstMatData[0], rows, 3*cols);
        break;
      default:
        cout << "ERR: Invalid image type! " << type2str(srcMat.type()) << endl;
        return;
      }

      dst = dstMat;
    }
  };

  BR_REGISTER(Transform, CUDACopyTo);
}

#include "cuda/copyto.moc"
