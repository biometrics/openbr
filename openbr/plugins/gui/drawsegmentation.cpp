#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace std;
using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Fill in the segmentations or draw a line between intersecting segments.
 * \author Austin Blanton \cite imaus10
 */
class DrawSegmentation : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(bool fillSegment READ get_fillSegment WRITE set_fillSegment RESET reset_fillSegment STORED false)
    BR_PROPERTY(bool, fillSegment, true)

    void project(const Template &src, Template &dst) const
    {
        if (!src.file.contains("SegmentsMask") || !src.file.contains("NumSegments")) qFatal("Must supply a Contours object in the metadata to drawContours.");
        Mat segments = src.file.get<Mat>("SegmentsMask");
        int numSegments = src.file.get<int>("NumSegments");

        dst.file = src.file;
        Mat drawn = fillSegment ? Mat(segments.size(), CV_8UC3, Scalar::all(0)) : src.m();

        for (int i=1; i<numSegments+1; i++) {
            Mat mask = segments == i;
            if (fillSegment) { // color the whole segment
                // set to a random color - get ready for a craaaazy acid trip
                int b = theRNG().uniform(0, 255);
                int g = theRNG().uniform(0, 255);
                int r = theRNG().uniform(0, 255);
                drawn.setTo(Scalar(r,g,b), mask);
            } else { // draw lines where there's a color change
                vector<vector<Point> > contours;
                Scalar color(0,255,0);
                findContours(mask, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
                drawContours(drawn, contours, -1, color);
            }
        }

        dst.m() = drawn;
    }
};

BR_REGISTER(Transform, DrawSegmentation)

} // namespace br

#include "drawsegmentation.moc"
