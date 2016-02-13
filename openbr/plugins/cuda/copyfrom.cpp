#include <iostream>

#include <opencv2/opencv.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace std;

using namespace cv;

// extern CUDA declaration
namespace br { namespace cuda { namespace cudacopyfrom {
  //template <typename T> void wrapper(void* src, T* out, int rows, int cols) {
  void wrapper(void* src, float* out, const int rows, const int cols);
}}}

namespace br
{
  class CUDACopyFrom : public UntrainableTransform
  {
    Q_OBJECT

private:
    void project(const Template &src, Template &dst) const
    {
      cout << "CUDACopyFrom Start" << endl << endl << endl;

      // pull the data back out of the Mat
      void* const* dataPtr = src.m().ptr<void*>();
      void* cudaMemPtr = dataPtr[0];
      int rows = *((int*)dataPtr[1]);
      int cols = *((int*)dataPtr[2]);
      int type = *((int*)dataPtr[3]);

      if (type != CV_32FC1) {
        cout << "ERR: Invalid data type!" << endl;
        return;
      }

      cout << "cudaMemPtr: " << cudaMemPtr << endl;
      cout << "rows: " << rows << endl;
      cout << "cols: " << cols << endl;
      cout << "type: " << type << endl;

      Mat dstMat = Mat(rows, cols, type);
      br::cuda::cudacopyfrom::wrapper(cudaMemPtr, dstMat.ptr<float>(), rows, cols);
      dst = dstMat;

      cout << "CUDACopyFrom End" << endl;

      cout << "DST Data" << endl;
      cout << "rows: " << dstMat.rows << endl;
      cout << "cols: " << dstMat.cols << endl;
      cout << "type: " << dstMat.type() << endl;
    }
  };

  BR_REGISTER(Transform, CUDACopyFrom);
}

#include "cuda/copyfrom.moc"
