#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief http://worldofcameras.wordpress.com/tag/skin-detection-opencv/
 * \author Josh Klontz \cite jklontz
 */
class SkinMaskTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        Mat m;
        cvtColor(src, m, CV_BGR2YCrCb);
        std::vector<Mat> mv;
        split(m, mv);
        Mat mask = Mat(m.rows, m.cols, CV_8UC1);

        for (int i=0; i<m.rows; i++) {
            for (int j=0; j<m.cols; j++) {
                int Cr= mv[1].at<quint8>(i,j);
                int Cb =mv[2].at<quint8>(i,j);
                mask.at<quint8>(i, j) = (Cr>130 && Cr<170) && (Cb>70 && Cb<125) ? 255 : 0;
            }
        }

        dst = mask;
    }
};

BR_REGISTER(Transform, SkinMaskTransform)

} // namespace br

#include "imgproc/skinmask.moc"
