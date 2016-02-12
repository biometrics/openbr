#include <iostream>

#include <opencv2/opencv.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace std;

using namespace cv;

// extern CUDA declaration
namespace br { namespace cuda { namespace cudacopyfrom {
  //template <typename T> void wrapper(void* src, T* out, int rows, int cols) {
  void wrapper(void* src, unsigned char* out, const int rows, const int cols);
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
      void* cudaMemPtr = dataPtr[0];
      int rows = *((int*)dataPtr[1]);
      int cols = *((int*)dataPtr[2]);
      int type = *((int*)dataPtr[3]);

      dst = Mat(rows, cols, type);

      br::cuda::cudacopyfrom::wrapper(cudaMemPtr, dst.m().ptr<unsigned char>(), rows, cols);
    }
  };

  BR_REGISTER(Transform, CUDACopyFrom);
}

#include "cuda/copyfrom.moc"
