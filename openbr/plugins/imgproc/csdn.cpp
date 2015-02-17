#include <openbr/plugins/openbr_internal.h>

using namespace std;
using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Meyers, E.; Wolf, L.
 * “Using biologically inspired features for face processing,”
 * Int. Journal of Computer Vision, vol. 76, no. 1, pp 93–104, 2008.
 * \author Scott Klum \cite sklum
 */

class CSDNTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(float s READ get_s WRITE set_s RESET reset_s STORED false)
    BR_PROPERTY(int, s, 16)

    void project(const Template &src, Template &dst) const
    {
        if (src.m().channels() != 1) qFatal("Expected single channel source matrix.");

        const int nRows = src.m().rows;
        const int nCols = src.m().cols;

        Mat m;
        src.m().convertTo(m, CV_32FC1);

        const int surround = s/2;

        for ( int i = 0; i < nRows; i++ ) {
            for ( int j = 0; j < nCols; j++ ) {
                int width = min( j+surround, nCols ) - max( 0, j-surround );
                int height = min( i+surround, nRows ) - max( 0, i-surround );

                Rect_<int> ROI(max(0, j-surround), max(0, i-surround), width, height);

                Scalar_<float> avg = mean(m(ROI));

                m.at<float>(i,j) = m.at<float>(i,j) - avg[0];
            }
        }

        dst = m;

    }
};

BR_REGISTER(Transform, CSDNTransform)

} // namespace br

#include "imgproc/csdn.moc"
