#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{
  class CUDAPassthroughTransform : public UntrainableTransform
  {
    Q_OBJECT

private:
    void project(const Template &src, Template &dst) const
    {
      dst = src;
    }
  };

  BR_REGISTER(Transform, CUDAPassthroughTransform);
}

#include "cuda/passthrough.moc"
