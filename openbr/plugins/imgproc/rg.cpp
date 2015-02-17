#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Normalized RG color space.
 * \author Josh Klontz \cite jklontz
 */
class RGTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        if (src.m().type() != CV_8UC3)
            qFatal("Expected CV_8UC3 images.");

        const Mat &m = src.m();
        Mat R(m.size(), CV_8UC1); // R / (R+G+B)
        Mat G(m.size(), CV_8UC1); // G / (R+G+B)

        for (int i=0; i<m.rows; i++)
            for (int j=0; j<m.cols; j++) {
                Vec3b v = m.at<Vec3b>(i,j);
                const int b = v[0];
                const int g = v[1];
                const int r = v[2];
                const int sum = b + g + r;
                if (sum > 0) {
                    R.at<uchar>(i, j) = saturate_cast<uchar>(255.0*r/(r+g+b));
                    G.at<uchar>(i, j) = saturate_cast<uchar>(255.0*g/(r+g+b));
                } else {
                    R.at<uchar>(i, j) = 0;
                    G.at<uchar>(i, j) = 0;
                }
            }

        dst.append(R);
        dst.append(G);
    }
};

BR_REGISTER(Transform, RGTransform)

} // namespace br

#include "imgproc/rg.moc"
