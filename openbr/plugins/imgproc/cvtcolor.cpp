#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Colorspace conversion.
 * \author Josh Klontz \cite jklontz
 */
class CvtColorTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_ENUMS(ColorSpace)
    Q_PROPERTY(ColorSpace colorSpace READ get_colorSpace WRITE set_colorSpace RESET reset_colorSpace STORED false)
    Q_PROPERTY(int channel READ get_channel WRITE set_channel RESET reset_channel STORED false)

public:
    enum ColorSpace { Gray = CV_BGR2GRAY,
                      RGBGray = CV_RGB2GRAY,
                      HLS = CV_BGR2HLS,
                      HSV = CV_BGR2HSV,
                      Lab = CV_BGR2Lab,
                      Luv = CV_BGR2Luv,
                      RGB = CV_BGR2RGB,
                      XYZ = CV_BGR2XYZ,
                      YCrCb = CV_BGR2YCrCb,
                      Color = CV_GRAY2BGR };

private:
    BR_PROPERTY(ColorSpace, colorSpace, Gray)
    BR_PROPERTY(int, channel, -1)

    void project(const Template &src, Template &dst) const
    {
        if (src.m().channels() > 1 || colorSpace == CV_GRAY2BGR) cvtColor(src, dst, colorSpace);
        else dst = src;

        if (channel != -1) {
            std::vector<Mat> mv;
            split(dst, mv);
            dst = mv[channel % (int)mv.size()];
        }
    }
};

BR_REGISTER(Transform, CvtColorTransform)

} // namespace br

#include "imgproc/cvtcolor.moc"
