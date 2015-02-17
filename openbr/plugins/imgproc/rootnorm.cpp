#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief dst=sqrt(norm_L1(src)) proposed as RootSIFT in \cite Arandjelovic12
 * \author Josh Klontz \cite jklontz
 */
class RootNormTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        const Mat &m = src;
        dst.m() = Mat(m.rows, m.cols, m.type());
        for (int i=0; i<m.rows; i++) {
            Mat temp;
            cv::normalize(m.row(i), temp, 1, 0, NORM_L1);
            cv::sqrt(temp, temp);
            temp.copyTo(dst.m().row(i));
        }
    }
};

BR_REGISTER(Transform, RootNormTransform)

} // namespace br

#include "imgproc/rootnorm.moc"
