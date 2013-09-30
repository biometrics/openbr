#include <opencv2/imgproc/imgproc.hpp>

#include "openbr_internal.h"

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Face Recognition Using Early Biologically Inspired Features
 * Min Li (IBM China Research Lab, China), Nalini Ratha (IBM Watson Research Center,
 * USA), Weihong Qian (IBM China Research Lab, China), Shenghua Bao (IBM China
 * Research Lab, China), Zhong Su (IBM China Research Lab, China)
 * \author Josh Klontz \cite jklontz
 */

class EBIFTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int N READ get_N WRITE set_N RESET reset_N STORED false) // scales
    Q_PROPERTY(int M READ get_M WRITE set_M RESET reset_M STORED false) // orientations
    BR_PROPERTY(int, N, 6)
    BR_PROPERTY(int, M, 9)

    void project(const Template &src, Template &dst) const
    {
        // Compute the image pyramid
        QList<Mat> scales;
        float scaleFactor = 1;
        for (int n=0; n<N; n++) {
            Mat scale;
            const int width = src.m().cols/scaleFactor;
            const int height = src.m().rows/scaleFactor;
            resize(src, scale, Size(width, height));
            scales.append(scale);
            scaleFactor /= sqrt(2.f);
        }

        (void) dst;
    }
};

BR_REGISTER(Transform, EBIFTransform)

} // namespace br

#include "ebif.moc"
