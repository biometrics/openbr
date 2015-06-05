/*
# PyVision License
#
# Copyright (c) 2006-2008 David S. Bolme
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither name of copyright holders nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Find eye locations using an ASEF filter
 * \br_paper Bolme, D.S.; Draper, B.A.; Beveridge, J.R.;
 *           "Average of Synthetic Exact Filters,"
 *           Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on , vol., no., pp.2105-2112, 20-25 June 2009
 * \author Josh Klontz \cite jklontz
 */
class ASEFEyesTransform : public UntrainableTransform
{
    Q_OBJECT

    Mat left_filter_dft, right_filter_dft, lut;
    Rect left_rect, right_rect;
    int width, height;

public:
    ASEFEyesTransform()
    {
        QFile file;
        QByteArray line, lf, rf, magic_number;
        QList<QByteArray> words;
        int r, c;
        Scalar t1, t2;

        // Open the eye locator model
        file.setFileName(Globals->sdkPath + "/share/openbr/models/EyeLocatorASEF128x128.fel");
        if (!file.open(QFile::ReadOnly)) qFatal("Failed to open %s for reading.", qPrintable(file.fileName()));

        // Check the first line
        if (file.readLine().simplified() != "CFEL") qFatal("Invalid header.");

        // Read past the comment and copyright.
        file.readLine();
        file.readLine();

        // Get the width and the height
        words = file.readLine().simplified().split(' ');
        r = words[0].toInt();
        c = words[1].toInt();

        // Read in the left bounding rectangle
        words = file.readLine().simplified().split(' ');
        left_rect = Rect(words[0].toInt(), words[1].toInt(), words[2].toInt(), words[3].toInt());

        // Read in the right bounding rectangle
        words = file.readLine().simplified().split(' ');
        right_rect = Rect(words[0].toInt(), words[1].toInt(), words[2].toInt(), words[3].toInt());

        // Read the magic number
        magic_number = file.readLine().simplified();

        // Read in the filter data
        lf = file.read(4*r*c);
        rf = file.read(4*r*c);
        file.close();

        // Test the magic number and byteswap if necessary.
        if (magic_number == "ABCD") {
            // Do nothing
        } else if (magic_number == "DCBA") {
            // Reverse the endianness
            // No swapping needed, not sure why
        } else {
            qFatal("Invalid Magic Number");
        }

        // Create the left and right filters
        Mat left_mat  = Mat(r, c, CV_32F, lf.data());
        Mat right_mat = Mat(r, c, CV_32F, rf.data());

        Mat left_filter;
        meanStdDev(left_mat, t1, t2);
        left_mat.convertTo(left_filter, -1, 1.0/t2[0], -t1[0]*1.0/t2[0]);

        Mat right_filter;
        meanStdDev(right_mat, t1, t2);
        right_mat.convertTo(right_filter, -1, 1.0/t2[0], -t1[0]*1.0/t2[0]);

        // Check the input to this function
        height = left_filter.rows;
        width = left_filter.cols;
        assert((left_filter.rows == right_filter.rows) &&
               (left_filter.cols == right_filter.cols) &&
               (left_filter.channels() == 1) &&
               (right_filter.channels() == 1));

        // Create the arrays needed for the computation
        left_filter_dft  = Mat(r, c, CV_32F);
        right_filter_dft = Mat(r, c, CV_32F);

        // Compute the filters in the Fourier domain
        dft(left_filter, left_filter_dft, CV_DXT_FORWARD);
        dft(right_filter, right_filter_dft, CV_DXT_FORWARD);

        // Create the look up table for the log transform
        lut = Mat(256, 1, CV_32F);
        for (int i=0; i<256; i++) lut.at<float>(i, 0) = std::log((float)i+1);
    }

private:
    void project(const Template &src, Template &dst) const
    {
        Rect roi = OpenCVUtils::toRect(src.file.rects().first());

        Mat gray;
        OpenCVUtils::cvtGray(src.m()(roi), gray);

        Mat image_tile;
        // (r,c) == (128, 128) EyeLocatorASEF128x128.fel
        resize(gray, image_tile, Size(height, width));

        // _preprocess
        Mat image;
        LUT(image_tile, lut, image);

        // correlate
        Mat left_corr, right_corr;
        dft(image, image, CV_DXT_FORWARD);
        mulSpectrums(image, left_filter_dft, left_corr, 0, true);
        mulSpectrums(image, right_filter_dft, right_corr, 0, true);
        dft(left_corr, left_corr, CV_DXT_INV_SCALE);
        dft(right_corr, right_corr, CV_DXT_INV_SCALE);

        // locateEyes
        double minVal, maxVal;
        Point minLoc, maxLoc;

        // left_rect == (23, 35)  (32, 32) EyeLocatorASEF128x128.fel
        minMaxLoc(left_corr(left_rect), &minVal, &maxVal, &minLoc, &maxLoc);
        float first_eye_x = (left_rect.x + maxLoc.x)*gray.cols/width+roi.x;
        float first_eye_y = (left_rect.y + maxLoc.y)*gray.rows/height+roi.y;

        // right_rect == (71, 32)  (32, 32) EyeLocatorASEF128x128.fel
        minMaxLoc(right_corr(right_rect), &minVal, &maxVal, &minLoc, &maxLoc);
        float second_eye_x = (right_rect.x + maxLoc.x)*gray.cols/width+roi.x;
        float second_eye_y = (right_rect.y + maxLoc.y)*gray.rows/height+roi.y;

        dst.m() = src.m();
        dst.file.appendPoint(QPointF(first_eye_x, first_eye_y));
        dst.file.appendPoint(QPointF(second_eye_x, second_eye_y));
        dst.file.set("First_Eye", QPointF(first_eye_x, first_eye_y));
        dst.file.set("Second_Eye", QPointF(second_eye_x, second_eye_y));
    }
};

BR_REGISTER(Transform, ASEFEyesTransform)

} // namespace br

#include "metadata/eyes.moc"
