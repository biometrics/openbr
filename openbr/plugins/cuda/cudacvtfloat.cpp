#include <iostream>
using namespace std;
#include <unistd.h>

#include <opencv2/opencv.hpp>
using namespace cv;

#include <openbr/plugins/openbr_internal.h>

namespace br { namespace cuda { namespace cudacvtfloat {
  void wrapper(void* src, void** dst, int rows, int cols);
}}}

namespace br
{

/*!
 * \ingroup transforms
 * \brief Converts byte to floating point
 * \author Colin Heinzmann \cite DepthDeluxe
 */
class CUDACvtFloatTransform : public UntrainableTransform
{
    Q_OBJECT

  public:
    void project(const Template &src, Template &dst) const
    {
      void* const* srcDataPtr = src.m().ptr<void*>();
      int rows = *((int*)srcDataPtr[1]);
      int cols = *((int*)srcDataPtr[2]);
      int type = *((int*)srcDataPtr[3]);

      // assume the image type is 256-monochrome
      // TODO(colin): real exception handling
      if (type != CV_8UC1) {
        cout << "ERR: Invalid memory format" << endl;
        return;
      }

      // build the destination mat
      Mat dstMat = Mat(src.m().rows, src.m().cols, src.m().type());
      void** dstDataPtr = dstMat.ptr<void*>();
      dstDataPtr[1] = srcDataPtr[1];
      dstDataPtr[2] = srcDataPtr[2];
      dstDataPtr[3] = srcDataPtr[3]; *((int*)dstDataPtr[3]) = CV_32FC1;

      br::cuda::cudacvtfloat::wrapper(srcDataPtr[0], &dstDataPtr[0], rows, cols);
      dst = dstMat;
    }
};

BR_REGISTER(Transform, CUDACvtFloatTransform)

} // namespace br

#include "cuda/cudacvtfloat.moc"
