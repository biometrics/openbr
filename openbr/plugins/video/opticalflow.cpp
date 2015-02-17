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

#include <opencv2/video/tracking.hpp>
#include <opencv2/video/background_segm.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

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
    Q_PROPERTY(bool useMagnitude READ get_useMagnitude WRITE set_useMagnitude RESET reset_useMagnitude STORED false)
    // these defaults are optimized for KTH
    BR_PROPERTY(double, pyr_scale, 0.1)
    BR_PROPERTY(int, levels, 1)
    BR_PROPERTY(int, winsize, 5)
    BR_PROPERTY(int, iterations, 10)
    BR_PROPERTY(int, poly_n, 7)
    BR_PROPERTY(double, poly_sigma, 1.1)
    BR_PROPERTY(int, flags, 0)
    BR_PROPERTY(bool, useMagnitude, true)

    void project(const Template &src, Template &dst) const
    {
        // get the two images put there by AggregateFrames
        if (src.size() != 2) qFatal("Optical Flow requires two images.");
        Mat prevImg = src[0], nextImg = src[1], flow;
        if (src[0].channels() != 1) OpenCVUtils::cvtGray(src[0], prevImg);
        if (src[1].channels() != 1) OpenCVUtils::cvtGray(src[1], nextImg);
        calcOpticalFlowFarneback(prevImg, nextImg, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags);

        if (useMagnitude) {
            // the result is two channels
            Mat flowOneCh;
            std::vector<Mat> channels(2);
            split(flow, channels);
            magnitude(channels[0], channels[1], flowOneCh);
            dst += flowOneCh;
        } else {
            dst += flow;
        }
        dst.file = src.file;
    }
};

BR_REGISTER(Transform, OpticalFlowTransform)

} // namespace br

#include "video/opticalflow.moc"
