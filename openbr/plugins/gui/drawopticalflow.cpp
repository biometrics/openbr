#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Draw a line representing the direction and magnitude of optical flow at the specified points.
 * \author Austin Blanton \cite imaus10
 */
class DrawOpticalFlow : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(QString original READ get_original WRITE set_original RESET reset_original STORED false)
    BR_PROPERTY(QString, original, "original")

    void project(const Template &src, Template &dst) const
    {
        const Scalar color(0,255,0);
        Mat flow = src.m();
        dst = src;
        if (!dst.file.contains(original)) qFatal("The original img must be saved in the metadata with SaveMat.");
        dst.m() = dst.file.get<Mat>(original);
        dst.file.remove(original);
        foreach (const Point2f &pt, OpenCVUtils::toPoints(dst.file.points())) {
            Point2f dxy = flow.at<Point2f>(pt.y, pt.x);
            Point2f newPt(pt.x+dxy.x, pt.y+dxy.y);
            line(dst, pt, newPt, color);
        }
    }
};

BR_REGISTER(Transform, DrawOpticalFlow)

} // namespace br

#include "gui/drawopticalflow.moc"
