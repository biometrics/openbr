#include <opencv2/video/tracking.hpp>
#include <opencv2/video/background_segm.hpp>
#include "openbr_internal.h"
#include "openbr/core/opencvutils.h"

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Gets a one-channel dense optical flow from two images
 * \author Austin Blanton \cite imaus10
 */
class OpticalFlowTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(double pyr_scale READ get_pyr_scale WRITE set_pyr_scale RESET reset_pyr_scale STORED false)
    Q_PROPERTY(int levels READ get_levels WRITE set_levels RESET reset_levels STORED false)
    Q_PROPERTY(int winsize READ get_winsize WRITE set_winsize RESET reset_winsize STORED false)
    Q_PROPERTY(int iterations READ get_iterations WRITE set_iterations RESET reset_iterations STORED false)
    Q_PROPERTY(int poly_n READ get_poly_n WRITE set_poly_n RESET reset_poly_n STORED false)
    Q_PROPERTY(double poly_sigma READ get_poly_sigma WRITE set_poly_sigma RESET reset_poly_sigma STORED false)
    Q_PROPERTY(int flags READ get_flags WRITE set_flags RESET reset_flags STORED false)
    // these defaults are optimized for KTH
    BR_PROPERTY(double, pyr_scale, 0.1)
    BR_PROPERTY(int, levels, 1)
    BR_PROPERTY(int, winsize, 5)
    BR_PROPERTY(int, iterations, 10)
    BR_PROPERTY(int, poly_n, 7)
    BR_PROPERTY(double, poly_sigma, 1.1)
    BR_PROPERTY(int, flags, 0)

    void project(const Template &src, Template &dst) const
    {
        // get the two images put there by AggregateFrames
        if (src.size() != 2) qFatal("Optical Flow requires two images.");
        Mat prevImg = src[0], nextImg = src[1], flow, flowOneCh;
        if (src[0].channels() != 1) OpenCVUtils::cvtGray(src[0], prevImg);
        if (src[1].channels() != 1) OpenCVUtils::cvtGray(src[1], nextImg);
        calcOpticalFlowFarneback(prevImg, nextImg, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags);

        // the result is two channels
        std::vector<Mat> channels(2);
        split(flow, channels);
        magnitude(channels[0], channels[1], flowOneCh);

        dst += flowOneCh;
        dst.file = src.file;
    }
};

BR_REGISTER(Transform, OpticalFlowTransform)

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV's BackgroundSubtractorMOG2 and puts the foreground mask in the Template metadata.
 * \author Austin Blanton \cite imaus10
 */
class SubtractBackgroundTransform : public TimeVaryingTransform
{
    Q_OBJECT

    BackgroundSubtractorMOG2 mog;

public:
    SubtractBackgroundTransform() : TimeVaryingTransform(false, false) {}

private:
    void projectUpdate(const Template &src, Template &dst)
    {
        dst = src;
        Mat mask;
        mog(src, mask);
        dst.file.set("Mask", QVariant::fromValue(mask));
    }

    void project(const Template &src, Template &dst) const
    {
        (void) src; (void) dst; qFatal("no way");
    }

    void finalize(TemplateList &output)
    {
        (void) output;
        mog = BackgroundSubtractorMOG2();
    }
};

BR_REGISTER(Transform, SubtractBackgroundTransform)

} // namespace br

#include "motion.moc"
