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

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include "openbr_internal.h"
#include "openbr/core/opencvutils.h"

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Renders metadata onto the image.
 *
 * The inPlace argument controls whether or not the image is cloned before the metadata is drawn.
 *
 * \author Josh Klontz \cite jklontz
 */
class DrawTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(bool verbose READ get_verbose WRITE set_verbose RESET reset_verbose STORED false)
    Q_PROPERTY(bool points READ get_points WRITE set_points RESET reset_points STORED false)
    Q_PROPERTY(bool rects READ get_rects WRITE set_rects RESET reset_rects STORED false)
    Q_PROPERTY(bool inPlace READ get_inPlace WRITE set_inPlace RESET reset_inPlace STORED false)
    Q_PROPERTY(int lineThickness READ get_lineThickness WRITE set_lineThickness RESET reset_lineThickness STORED false)
    BR_PROPERTY(bool, verbose, false)
    BR_PROPERTY(bool, points, true)
    BR_PROPERTY(bool, rects, true)
    BR_PROPERTY(bool, inPlace, false)
    BR_PROPERTY(int, lineThickness, 1)

    void project(const Template &src, Template &dst) const
    {
        const Scalar color(0,255,0);
        const Scalar verboseColor(255, 255, 0);
        dst.m() = inPlace ? src.m() : src.m().clone();

        if (points) {
            const QList<Point2f> pointsList = OpenCVUtils::toPoints(src.file.namedPoints() + src.file.points());
            for (int i=0; i<pointsList.size(); i++) {
                const Point2f &point = pointsList[i];
                circle(dst, point, 3, color, -1);
                if (verbose) putText(dst, QString("%1,(%2,%3)").arg(QString::number(i),QString::number(point.x),QString::number(point.y)).toStdString(), point, FONT_HERSHEY_SIMPLEX, 0.5, verboseColor, 1);
            }
        }
        if (rects) {
            foreach (const Rect &rect, OpenCVUtils::toRects(src.file.namedRects() + src.file.rects()))
                rectangle(dst, rect, color, lineThickness);
        }
    }
};

BR_REGISTER(Transform, DrawTransform)


/*!
 * \ingroup transforms
 * \brief Draw the value of the specified property at the specified point on the image
 *
 * The inPlace argument controls whether or not the image is cloned before it is drawn on.
 *
 * \author Charles Otto \cite caotto
 */
class DrawPropertyPointTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(QString propName READ get_propName WRITE set_propName RESET reset_propName STORED false)
    Q_PROPERTY(QString pointName READ get_pointName WRITE set_pointName RESET reset_pointName STORED false)
    Q_PROPERTY(bool inPlace READ get_inPlace WRITE set_inPlace RESET reset_inPlace STORED false)
    BR_PROPERTY(QString, propName, "")
    BR_PROPERTY(QString, pointName, "")
    BR_PROPERTY(bool, inPlace, false)


    void project(const Template &src, Template &dst) const
    {
        dst = src;
        if (propName.isEmpty() || pointName.isEmpty())
            return;

        dst.m() = inPlace ? src.m() : src.m().clone();

        const Scalar textColor(255, 255, 0);

        QVariant prop = dst.file.value(propName);


        if (!prop.canConvert(QVariant::String))
            return;
        QString propString = prop.toString();

        QVariant point = dst.file.value(pointName);

        if (!point.canConvert(QVariant::PointF))
            return;

        QPointF targetPoint = point.toPointF();

        Point2f cvPoint =OpenCVUtils::toPoint(targetPoint);

        std::string text = propName.toStdString() + ": " + propString.toStdString();
        putText(dst, text, cvPoint, FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1);
    }

};
BR_REGISTER(Transform, DrawPropertyPointTransform)

/*!
 * \ingroup transforms
 * \brief Draw the values of a list of properties at the specified point on the image
 *
 * The inPlace argument controls whether or not the image is cloned before it is drawn on.
 *
 * \author Charles Otto \cite caotto
 */
class DrawPropertiesPointTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(QStringList propNames READ get_propNames WRITE set_propNames RESET reset_propNames STORED false)
    Q_PROPERTY(QString pointName READ get_pointName WRITE set_pointName RESET reset_pointName STORED false)
    Q_PROPERTY(bool inPlace READ get_inPlace WRITE set_inPlace RESET reset_inPlace STORED false)
    BR_PROPERTY(QStringList, propNames, QStringList())
    BR_PROPERTY(QString, pointName, "")
    BR_PROPERTY(bool, inPlace, false)

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        if (propNames.isEmpty() || pointName.isEmpty())
            return;

        dst.m() = inPlace ? src.m() : src.m().clone();

        QVariant point = dst.file.value(pointName);

        if (!point.canConvert(QVariant::PointF))
            return;

        QPointF targetPoint = point.toPointF();

        Point2f cvPoint =OpenCVUtils::toPoint(targetPoint);


        const Scalar textColor(255, 255, 0);

        std::string outString = "";
        foreach (const QString &propName, propNames)
        {
            QVariant prop = dst.file.value(propName);

            if (!prop.canConvert(QVariant::String))
                continue;
            QString propString = prop.toString();
            outString += propName.toStdString() + ": " + propString.toStdString() + " ";

        }
        if (outString.empty())
            return;

        putText(dst, outString, cvPoint, FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1);
    }

};
BR_REGISTER(Transform, DrawPropertiesPointTransform)


/*!
 * \ingroup transforms
 * \brief Draws a grid on the image
 * \author Josh Klontz \cite jklontz
 */
class DrawGridTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int rows READ get_rows WRITE set_rows RESET reset_rows STORED false)
    Q_PROPERTY(int columns READ get_columns WRITE set_columns RESET reset_columns STORED false)
    Q_PROPERTY(int r READ get_r WRITE set_r RESET reset_r STORED false)
    Q_PROPERTY(int g READ get_g WRITE set_g RESET reset_g STORED false)
    Q_PROPERTY(int b READ get_b WRITE set_b RESET reset_b STORED false)
    BR_PROPERTY(int, rows, 0)
    BR_PROPERTY(int, columns, 0)
    BR_PROPERTY(int, r, 196)
    BR_PROPERTY(int, g, 196)
    BR_PROPERTY(int, b, 196)

    void project(const Template &src, Template &dst) const
    {
        Mat m = src.m().clone();
        float rowStep = 1.f * m.rows / (rows+1);
        float columnStep = 1.f * m.cols / (columns+1);
        int thickness = qMin(m.rows, m.cols) / 256;
        for (float row = rowStep/2; row < m.rows; row += rowStep)
            line(m, Point(0, row), Point(m.cols, row), Scalar(r, g, b), thickness, CV_AA);
        for (float column = columnStep/2; column < m.cols; column += columnStep)
            line(m, Point(column, 0), Point(column, m.rows), Scalar(r, g, b), thickness, CV_AA);
        dst = m;
    }
};

BR_REGISTER(Transform, DrawGridTransform)

/*!
 * \ingroup transforms
 * \brief Computes the mean of a set of templates.
 * \note Suitable for visualization only as it sets every projected template to the mean template.
 * \author Scott Klum \cite sklum
 */
class MeanTransform : public Transform
{
    Q_OBJECT

    Mat mean;

    void train(const TemplateList &data)
    {
        mean = Mat::zeros(data[0].m().rows,data[0].m().cols,CV_32F);

        for (int i = 0; i < data.size(); i++) {
            Mat converted;
            data[i].m().convertTo(converted, CV_32F);
            mean += converted;
        }

        mean /= data.size();
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        dst.m() = mean;
    }

};

BR_REGISTER(Transform, MeanTransform)

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
            clip_roi.height -= abs(outIm.rows - (clip_roi.height + target_location.top() ));
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

/*!
 * \ingroup transforms
 * \brief Write all mats to disk as images.
 * \author Brendan Klare \bklare
 */
class WriteImageTransform : public TimeVaryingTransform
{
    Q_OBJECT
    Q_PROPERTY(QString outputDirectory READ get_outputDirectory WRITE set_outputDirectory RESET reset_outputDirectory STORED false)
    Q_PROPERTY(QString imageName READ get_imageName WRITE set_imageName RESET reset_imageName STORED false)
    Q_PROPERTY(QString imgExtension READ get_imgExtension WRITE set_imgExtension RESET reset_imgExtension STORED false)
    BR_PROPERTY(QString, outputDirectory, "Temp")
    BR_PROPERTY(QString, imageName, "image")
    BR_PROPERTY(QString, imgExtension, "jpg")

    int cnt;

    void init() {
        cnt = 0;
        if (! QDir(outputDirectory).exists())
            QDir().mkdir(outputDirectory);
    }

    void projectUpdate(const Template &src, Template &dst)
    {
        dst = src;
        OpenCVUtils::saveImage(dst.m(), QString("%1/%2_%3.%4").arg(outputDirectory).arg(imageName).arg(cnt++, 5, QChar('0')).arg(imgExtension));
    }

};
BR_REGISTER(Transform, WriteImageTransform)


/**
 * @brief The MeanImageTransform class computes the average template/image
 * and save the result as an encoded image.
 */
class MeanImageTransform : public TimeVaryingTransform
{
    Q_OBJECT

    Q_PROPERTY(QString imgname READ get_imgname WRITE set_imgname RESET reset_imgname STORED false)
    Q_PROPERTY(QString ext READ get_ext WRITE set_ext RESET reset_ext STORED false)

    BR_PROPERTY(QString, imgname, "average")
    BR_PROPERTY(QString, ext, "jpg")

    Mat average;
    int cnt;

    void init()
    {
        cnt = 0;
    }

    void projectUpdate(const Template &src, Template &dst)
    {
        dst = src;
        if (cnt == 0) {
            if (src.m().channels() == 1)
                average = Mat::zeros(dst.m().size(),CV_64FC1);
            else if (src.m().channels() == 3)
                average = Mat::zeros(dst.m().size(),CV_64FC3);
            else
                qFatal("Unsupported number of channels");
        }

        Mat temp;
        if (src.m().channels() == 1) {
            src.m().convertTo(temp, CV_64FC1);
            average += temp;
        } else if (src.m().channels() == 3) {
            src.m().convertTo(temp, CV_64FC3);
            average += temp;
        } else
            qFatal("Unsupported number of channels");

        cnt++;
    }

    virtual void finalize(TemplateList &output)
    {
        average /= float(cnt);
        imwrite(QString("%1.%2").arg(imgname).arg(ext).toStdString(), average);
        output = TemplateList();
    }


public:
    MeanImageTransform() : TimeVaryingTransform(false, false) {}
};

BR_REGISTER(Transform, MeanImageTransform)


// TODO: re-implement EditTransform using Qt
#if 0
/*!
 * \ingroup transforms
 * \brief Remove landmarks.
 * \author Josh Klontz \cite jklontz
 */
class EditTransform : public UntrainableTransform
{
    Q_OBJECT

    Transform *draw;
    static Template currentTemplate;
    static QMutex currentTemplateLock;

    void init()
    {
        draw = make("Draw");
        Globals->setProperty("parallelism", "0"); // Can only work in single threaded mode
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        if (Globals->parallelism) {
            qWarning("Edit::project() only works in single threaded mode.");
            return;
        }

        currentTemplateLock.lock();
        currentTemplate = src;
        OpenCVUtils::showImage(src, "Edit", false);
        setMouseCallback("Edit", mouseCallback, (void*)this);
        mouseEvent(0, 0, 0, 0);
        waitKey(-1);
        dst = currentTemplate;
        currentTemplateLock.unlock();
    }

    static void mouseCallback(int event, int x, int y, int flags, void *userdata)
    {
        ((const EditTransform*)userdata)->mouseEvent(event, x, y, flags);
    }

    void mouseEvent(int event, int x, int y, int flags) const
    {
        (void) event;
        if (flags) {
            QList<QRectF> rects = currentTemplate.file.rects();
            for (int i=rects.size()-1; i>=0; i--)
                if (rects[i].contains(x,y))
                    rects.removeAt(i);
            currentTemplate.file.setRects(rects);
        }

        Template temp;
        draw->project(currentTemplate, temp);
        OpenCVUtils::showImage(temp, "Edit", false);
    }
};

Template EditTransform::currentTemplate;
QMutex EditTransform::currentTemplateLock;

BR_REGISTER(Transform, EditTransform)
#endif

} // namespace br

#include "draw.moc"
