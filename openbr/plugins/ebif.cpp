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

    Transform *gaborJet;

    void init()
    {
        QStringList thetas; // Orientations between 0 and pi
        for (int m=0; m<M; m++)
            thetas.append(QString::number(CV_PI*m/M));
        gaborJet = make(QString("GaborJet([%1],[%2],[%3],[%4],[%5])").arg(
            QString::number(5), // lambda = 5 (just one wavelength)
            thetas.join(','), // M orientations between 0 and pi
            QString::number(0), // psi = 0 (no offset)
            QString::number(3), // sigma = 3 (just one width)
            QString::number(1) // gamma = 1 (no skew)
            ));
    }

    void project(const Template &src, Template &dst) const
    {
        // Compute the image pyramid
        Template scales;
        float scaleFactor = 1;
        for (int n=0; n<N; n++) {
            Mat scale;
            const int width = src.m().cols/scaleFactor;
            const int height = src.m().rows/scaleFactor;
            resize(src, scale, Size(width, height));
            scales.append(scale);
            scaleFactor /= sqrt(2.f);
        }

        // Perform gabor wavelet convolution on all scales
        scales >> *gaborJet;

        (void) dst;
    }
};

BR_REGISTER(Transform, EBIFTransform)

} // namespace br

#include "ebif.moc"
