#include <opencv2/video/tracking.hpp>
#include "openbr_internal.h"

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Gets a two-channel dense optical flow from two images
 * \author Austin Blanton \cite imaus10
 */
class OpticalFlowTransform : public UntrainableTransform
{
    Q_OBJECT
    // these defaults are optimized for KTH
    Q_PROPERTY(double pyr_scale READ get_pyr_scale WRITE set_pyr_scale RESET reset_pyr_scale STORE false)
    Q_PROPERTY(int levels READ get_levels WRITE set_levels RESET reset_levels STORE false)
    Q_PROPERTY(int winsize READ get_winsize WRITE set_winsize RESET reset_winsize STORE false)
    Q_PROPERTY(int iterations READ get_iterations WRITE set_iterations RESET reset_iterations STORE false)
    Q_PROPERTY(int poly_n READ get_poly_n WRITE set_poly_n RESET reset_poly_n STORE false)
    Q_PROPERTY(double poly_sigma READ get_poly_sigma WRITE set_poly_sigma RESET reset_poly_sigma STORE false)
    Q_PROPERTY(int flags READ get_flags WRITE set_flags RESET reset_flags STORE false)
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
        Mat prevImg = src[0];
        Mat nextImg = src[1];
        Mat flow;
        calcOpticalFlowFarneback(prevImg, nextImg, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags);
        dst += flow;
    }
};

BR_REGISTER(Transform, OpticalFlowTransform)

} // namespace br

#include "opticalflow.moc"
