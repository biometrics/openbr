#include <opencv2/imgproc/imgproc.hpp>
#include "openbr_internal.h"
#include "openbr/core/opencvutils.h"

using namespace cv;

namespace br
{

class SegmentationTransform : public UntrainableTransform
{
    Q_OBJECT
    void project(const Template &src, Template &dst) const
    {
        dst = src;

        Mat mod;
//        adaptiveThreshold(src.m(), src.m(), 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 33, 5);
        threshold(src.m(), mod, 0, 255, THRESH_BINARY+THRESH_OTSU);

        // findContours requires an 8-bit 1-channel image
        // and modifies its source image
        if (mod.depth() != CV_8U) OpenCVUtils::cvtUChar(mod, mod);
        if (mod.channels() != 1) OpenCVUtils::cvtGray(mod, mod);
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours(mod, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

        // draw the contour delineations as 1,2,3... for input to watershed
        Mat markers(mod.size(), CV_32S);
        Scalar::all(0);
        int compCount=0;
        for (int idx=0; idx>=0; idx=hierarchy[idx][0], compCount++) {
            drawContours(markers, contours, idx, Scalar::all(compCount+1), -1, 8, hierarchy, INT_MAX);
        }

        watershed(src.file.get<Mat>("original"), markers);
        dst.file.set("SegmentsMask", QVariant::fromValue(markers));
        dst.file.set("NumSegments", compCount);
    }
};
BR_REGISTER(Transform, SegmentationTransform)

} // namespace br

#include "segmentation.moc"
