#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Kernel hash
 * \author Josh Klontz \cite jklontz
 */
class KernelHashTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(uchar dimsIn READ get_dimsIn WRITE set_dimsIn RESET reset_dimsIn STORED false)
    Q_PROPERTY(uchar dimsOut READ get_dimsOut WRITE set_dimsOut RESET reset_dimsOut STORED false)
    BR_PROPERTY(uchar, dimsIn, 8)
    BR_PROPERTY(uchar, dimsOut, 7)

    void project(const Template &src, Template &dst) const
    {
        if (src.m().type() != CV_8UC1)
            qFatal("Expected 8UC1 input.");

        dst = cv::Mat::zeros(src.m().rows, src.m().cols, CV_8UC1);
        const uchar *srcData = src.m().data;
        uchar *dstData = dst.m().data;
        const int step = src.m().cols;
        for (int i=0; i<src.m().rows-1; i++)
            for (int j=0; j<src.m().cols-1; j++) {
                dstData[i*step+j] = (uint(pow(float(dimsIn),1.f))*srcData[i    *step+j]
                                   + uint(pow(float(dimsIn),2.f))*srcData[(i+1)*step+j]
                                   + uint(pow(float(dimsIn),0.f))*srcData[i    *step+(j+1)]
                                   /*+ uint(pow(float(dimsIn),0.f))*srcData[(i+1)*step+(j+1)]*/) % dimsOut;
            }
    }
};

BR_REGISTER(Transform, KernelHashTransform)

} // namespace br

#include "imgproc/kernelhash.moc"
