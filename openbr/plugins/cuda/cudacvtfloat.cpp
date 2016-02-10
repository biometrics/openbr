#include <iostream>
#include <unistd.h>
using namespace std;

#include <opencv2/opencv.hpp>
using namespace cv;

#include <openbr/plugins/openbr_internal.h>

#include "cudacvtfloat.hpp"

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
      // assume the image type is 256-monochrome
      // TODO(colin): real exception handling
      if (src.m().type() != CV_8UC1) {
        cout << "ERR: Invalid memory format" << endl;
        return;
      }


      int rows = src.m().rows;
      int cols = src.m().cols;

      dst = Mat(rows, cols, CV_32FC1);

      br::cuda::cudacvtfloat::wrapper((const unsigned char*)src.m().ptr<unsigned char>(), dst.m().ptr<float>(), rows, cols);
    }
};

BR_REGISTER(Transform, CUDACvtFloatTransform)

} // namespace br

#include "cuda/cudacvtfloat.moc"
