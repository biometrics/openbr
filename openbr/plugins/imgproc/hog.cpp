#include <opencv2/objdetect/objdetect.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief OpenCV HOGDescriptor wrapper
 * \author Austin Blanton \cite imaus10
 */
class HoGDescriptorTransform : public UntrainableTransform
{
    Q_OBJECT

    HOGDescriptor hog;

    void project(const Template &src, Template &dst) const
    {
        std::vector<float> descriptorVals;
        std::vector<Point> locations;
        Size winStride = Size(0,0);
        Size padding = Size(0,0);
        foreach (const Mat &rect, src) {
            hog.compute(rect, descriptorVals, winStride, padding, locations);
            Mat HoGFeats(descriptorVals, true);
            dst += HoGFeats;
        }
    }
};

BR_REGISTER(Transform, HoGDescriptorTransform)

} // namespace br

#include "imgproc/hog.moc"
