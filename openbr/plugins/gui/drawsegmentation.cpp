/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2012 The MITRE Corporation                                      *
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License");           *
 * you may not use this file except in compliance with the License.          *
 * You may obtain a copy of the License at                                   *
 *                                                                           *
 *     http://www.apache.org/licenses/LICENSE-2.0                            *
 *                                                                           *
 * Unless required by applicable law or agreed to in writing, software       *
 * distributed under the License is distributed on an "AS IS" BASIS,         *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 * See the License for the specific language governing permissions and       *
 * limitations under the License.                                            *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

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

#include "gui/drawsegmentation.moc"
