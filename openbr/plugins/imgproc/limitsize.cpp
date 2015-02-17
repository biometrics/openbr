#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Limit the size of the template
 * \author Josh Klontz \cite jklontz
 */
class LimitSizeTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int max READ get_max WRITE set_max RESET reset_max STORED false)
    BR_PROPERTY(int, max, -1)

    void project(const Template &src, Template &dst) const
    {
        const Mat &m = src;
        if (m.rows > m.cols)
            if (m.rows > max) resize(m, dst, Size(std::max(1, m.cols * max / m.rows), max));
            else              dst = m;
        else
            if (m.cols > max) resize(m, dst, Size(max, std::max(1, m.rows * max / m.cols)));
            else              dst = m;
    }
};

BR_REGISTER(Transform, LimitSizeTransform)

} // namespace br

#include "imgproc/limitsize.moc"
