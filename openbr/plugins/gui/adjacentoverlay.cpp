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

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Load the image named in the specified property, draw it on the current matrix adjacent to the rect specified in the other property.
 * \author Charles Otto \cite caotto
 */
class AdjacentOverlayTransform : public Transform
{
    Q_OBJECT

    Q_PROPERTY(QString imgName READ get_imgName WRITE set_imgName RESET reset_imgName STORED false)
    Q_PROPERTY(QString targetName READ get_targetName WRITE set_targetName RESET reset_targetName STORED false)
    BR_PROPERTY(QString, imgName, "")
    BR_PROPERTY(QString, targetName, "")

    QSharedPointer<Transform> opener;
    void project(const Template &src, Template &dst) const
    {
        dst = src;

        if (imgName.isEmpty() || targetName.isEmpty() || !dst.file.contains(imgName) || !dst.file.contains(targetName))
            return;

        QVariant temp = src.file.value(imgName);
        cv::Mat im;
        // is this a filename?
        if (temp.canConvert<QString>()) {
            QString im_name = temp.toString();
            Template temp_im;
            opener->project(File(im_name), temp_im);
            im = temp_im.m();
        }
        // a cv::Mat ?
        else if (temp.canConvert<cv::Mat>())
            im = src.file.get<cv::Mat>(imgName);
        else
            qDebug() << "Unrecognized property type " << imgName << "for" << src.file.name;

        // Location of detected face in source image
        QRectF target_location = src.file.get<QRectF>(targetName);

        // match width with target region
        qreal target_width = target_location.width();
        qreal current_width = im.cols;
        qreal current_height = im.rows;

        qreal aspect_ratio = current_height / current_width;
        qreal target_height = target_width * aspect_ratio;

        cv::resize(im, im, cv::Size(target_width, target_height));

        // ROI used to maybe crop the matched image
        cv::Rect clip_roi;
        clip_roi.x = 0;
        clip_roi.y = 0;
        clip_roi.width = im.cols;
        clip_roi.height= im.rows <= dst.m().rows ? im.rows : dst.m().rows;

        int half_width = src.m().cols / 2;
        int out_x = 0;

        // place in the source image we will copy the matched image to.
        cv::Rect target_roi;
        bool left_side = false;
        int width_adjust = 0;
        // Place left
        if (target_location.center().rx() > half_width) {
            out_x = target_location.left() - im.cols;
            if (out_x < 0) {
                width_adjust = abs(out_x);
                out_x = 0;
            }
            left_side = true;
        }
        // place right
        else {
            out_x = target_location.right();
            int high = out_x + im.cols;
            if (high >= src.m().cols) {
                width_adjust = abs(high - src.m().cols + 1);
            }
        }

        cv::Mat outIm;
        if (width_adjust)
        {
            outIm.create(dst.m().rows, dst.m().cols + width_adjust, CV_8UC3);
            memset(outIm.data, 127, outIm.rows * outIm.cols * outIm.channels());

            Rect temp;

            if (left_side)
                temp = Rect(abs(width_adjust), 0, dst.m().cols, dst.m().rows);

            else
                temp = Rect(0, 0, dst.m().cols, dst.m().rows);

            dst.m().copyTo(outIm(temp));

        }
        else
            outIm = dst.m();

        if (clip_roi.height + target_location.top() >= outIm.rows)
        {
            clip_roi.height -= abs(int(outIm.rows - (clip_roi.height + target_location.top())));
        }
        if (clip_roi.x + clip_roi.width >= im.cols) {
            clip_roi.width -= abs(im.cols - (clip_roi.x + clip_roi.width + 1));
            if (clip_roi.width < 0)
                clip_roi.width = 1;
        }

        if (clip_roi.y + clip_roi.height >= im.rows) {
            clip_roi.height -= abs(im.rows - (clip_roi.y + clip_roi.height + 1));
        }
        if (clip_roi.x < 0)
            clip_roi.x = 0;
        if (clip_roi.y < 0)
            clip_roi.y = 0;

        if (clip_roi.height < 0)
            clip_roi.height = 0;

        if (clip_roi.width < 0)
            clip_roi.width = 0;


        if (clip_roi.y + clip_roi.height >= im.rows)
        {
            qDebug() << "Bad clip y" << clip_roi.y + clip_roi.height << im.rows;
        }
        if (clip_roi.x + clip_roi.width >= im.cols)
        {
            qDebug() << "Bad clip x" << clip_roi.x + clip_roi.width << im.cols;
        }

        if (clip_roi.y < 0 || clip_roi.height < 0)
        {
            qDebug() << "bad clip y, low" << clip_roi.y << clip_roi.height;
            qFatal("die");
        }
        if (clip_roi.x < 0 || clip_roi.width < 0)
        {
            qDebug() << "bad clip x, low" << clip_roi.x << clip_roi.width;
            qFatal("die");
        }

        target_roi.x = out_x;
        target_roi.width = clip_roi.width;
        target_roi.y = target_location.top();
        target_roi.height = clip_roi.height;


        im = im(clip_roi);

        if (target_roi.x < 0 || target_roi.x >= outIm.cols)
        {
            qDebug() << "Bad xdim in targetROI!" << target_roi.x << " out im x: " << outIm.cols;
            qFatal("die");
        }

        if (target_roi.x + target_roi.width < 0 || (target_roi.x + target_roi.width) >= outIm.cols)
        {
            qDebug() << "Bad xdim in targetROI!" << target_roi.x + target_roi.width;
            qFatal("die");
        }

        if (target_roi.y < 0 || target_roi.y >= outIm.rows)
        {
            qDebug() << "Bad ydim in targetROI!" << target_roi.y;
            qFatal("die");
        }

        if ((target_roi.y + target_roi.height) < 0 || (target_roi.y + target_roi.height) > outIm.rows)
        {
            qDebug() << "Bad ydim in targetROI!" << target_roi.y + target_roi.height;
            qDebug() << "target_roi.y: " << target_roi.y << " height: " << target_roi.height;
            qFatal("die");
        }


        std::vector<cv::Mat> channels;
        cv::split(outIm, channels);

        std::vector<cv::Mat> patch_channels;
        cv::split(im, patch_channels);

        for (size_t i=0; i < channels.size(); i++)
        {
            cv::addWeighted(channels[i](target_roi), 0, patch_channels[i % patch_channels.size()], 1, 0,channels[i](target_roi));
        }
        cv::merge(channels, outIm);
        dst.m() = outIm;

    }

    void init()
    {
        opener = QSharedPointer<br::Transform>(br::Transform::make("Cache(Open)", NULL));
    }

};

BR_REGISTER(Transform, AdjacentOverlayTransform)

} // namespace br

#include "gui/adjacentoverlay.moc"
