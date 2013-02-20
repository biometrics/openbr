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
#include <openbr_plugin.h>

#include "core/opencvutils.h"

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Renders metadata onto the image
 * \author Josh Klontz \cite jklontz
 */
class DrawTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(bool verbose READ get_verbose WRITE set_verbose RESET reset_verbose STORED false)
    BR_PROPERTY(bool, verbose, false)

    void project(const Template &src, Template &dst) const
    {
        const Scalar color(0,255,0);
        const Scalar verboseColor(255, 255, 0);
        dst = src.m().clone();

        QList<Point2f> landmarks = OpenCVUtils::toPoints(src.file.landmarks());

        foreach (const Point2f &landmark, landmarks)
            circle(dst, landmark, 3, color, -1);
        QList<Point2f> namedLandmarks = OpenCVUtils::toPoints(src.file.namedLandmarks());
        foreach (const Point2f &landmark, namedLandmarks)
            circle(dst, landmark, 3, color);
        QList<Rect> ROIs = OpenCVUtils::toRects(src.file.ROIs());
        foreach (const Rect ROI, ROIs)
            rectangle(dst, ROI, color);

        if (verbose)
            for (int i=0; i<landmarks.size(); i++)
                putText(dst, QString::number(i).toStdString(), landmarks[i], FONT_HERSHEY_SIMPLEX, 0.5, verboseColor, 1);
    }
};

BR_REGISTER(Transform, DrawTransform)

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
            QList<QRectF> ROIs = currentTemplate.file.ROIs();
            for (int i=ROIs.size()-1; i>=0; i--)
                if (ROIs[i].contains(x,y))
                    ROIs.removeAt(i);
            currentTemplate.file.setROIs(ROIs);
        }

        Template temp;
        draw->project(currentTemplate, temp);
        OpenCVUtils::showImage(temp, "Edit", false);
    }
};

Template EditTransform::currentTemplate;
QMutex EditTransform::currentTemplateLock;

BR_REGISTER(Transform, EditTransform)

} // namespace br

#include "draw.moc"
