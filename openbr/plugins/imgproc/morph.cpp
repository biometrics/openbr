#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Morphological operator
 * \author Josh Klontz \cite jklontz
 */
class MorphTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_ENUMS(Op)
    Q_PROPERTY(Op op READ get_op WRITE set_op RESET reset_op STORED false)
    Q_PROPERTY(int radius READ get_radius WRITE set_radius RESET reset_radius STORED false)

public:
    /*!< */
    enum Op { Erode = MORPH_ERODE,
              Dilate = MORPH_DILATE,
              Open = MORPH_OPEN,
              Close = MORPH_CLOSE,
              Gradient = MORPH_GRADIENT,
              TopHat = MORPH_TOPHAT,
              BlackHat = MORPH_BLACKHAT };

private:
    BR_PROPERTY(Op, op, Close)
    BR_PROPERTY(int, radius, 1)

    Mat kernel;

    void init()
    {
        Mat kernel = Mat(radius, radius, CV_8UC1);
        kernel.setTo(255);
    }

    void project(const Template &src, Template &dst) const
    {
        morphologyEx(src, dst, op, kernel);
    }
};

BR_REGISTER(Transform, MorphTransform)

} // namespace br

#include "imgproc/morph.moc"
