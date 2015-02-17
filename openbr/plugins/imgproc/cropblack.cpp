#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Crop out black borders
 * \author Josh Klontz \cite jklontz
 */
class CropBlackTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        Mat gray;
        OpenCVUtils::cvtGray(src, gray);

        int xStart = 0;
        while (xStart < gray.cols) {
            if (mean(gray.col(xStart))[0] >= 1) break;
            xStart++;
        }

        int xEnd = gray.cols - 1;
        while (xEnd >= 0) {
            if (mean(gray.col(xEnd))[0] >= 1) break;
            xEnd--;
        }

        int yStart = 0;
        while (yStart < gray.rows) {
            if (mean(gray.col(yStart))[0] >= 1) break;
            yStart++;
        }

        int yEnd = gray.rows - 1;
        while (yEnd >= 0) {
            if (mean(gray.col(yEnd))[0] >= 1) break;
            yEnd--;
        }

        dst = src.m()(Rect(xStart, yStart, xEnd-xStart, yEnd-yStart));
    }
};

BR_REGISTER(Transform, CropBlackTransform)

} // namespace br

#include "imgproc/cropblack.moc"
