#include <opencv2/photo/photo.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV inpainting
 * \author Josh Klontz \cite jklontz
 */
class InpaintTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_ENUMS(Method)
    Q_PROPERTY(int radius READ get_radius WRITE set_radius RESET reset_radius STORED false)
    Q_PROPERTY(Method method READ get_method WRITE set_method RESET reset_method STORED false)

public:
    /*!< */
    enum Method { NavierStokes = INPAINT_NS,
                  Telea = INPAINT_TELEA };

private:
    BR_PROPERTY(int, radius, 1)
    BR_PROPERTY(Method, method, NavierStokes)
    Transform *cvtGray;

    void init()
    {
        cvtGray = make("Cvt(Gray)");
    }

    void project(const Template &src, Template &dst) const
    {
        inpaint(src, (*cvtGray)(src)<5, dst, radius, method);
    }
};

} // namespace br

#include "imgproc/inpaint.moc"
