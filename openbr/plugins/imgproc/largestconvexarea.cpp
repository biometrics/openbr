#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Set the template's label to the area of the largest convex hull.
 * \author Josh Klontz \cite jklontz
 */
class LargestConvexAreaTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(QString outputVariable READ get_outputVariable WRITE set_outputVariable RESET reset_outputVariable STORED false)
    BR_PROPERTY(QString, outputVariable, "Label")

    void project(const Template &src, Template &dst) const
    {
        std::vector< std::vector<Point> > contours;
        findContours(src.m().clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        double maxArea = 0;
        foreach (const std::vector<Point> &contour, contours) {
            std::vector<Point> hull;
            convexHull(contour, hull);
            double area = contourArea(contour);
            double hullArea = contourArea(hull);
            if (area / hullArea > 0.98)
                maxArea = std::max(maxArea, area);
        }
        dst.file.set(outputVariable, maxArea);
    }
};

BR_REGISTER(Transform, LargestConvexAreaTransform)

} // namespace br

#include "imgproc/largestconvexarea.moc"
