#include "faceviewer.h"

#include <QPainter>
#include <QtConcurrentRun>

#include <opencv2/opencv.hpp>

const int numLandmarks = 2;
const double proportionality = 0.025;

using namespace br;
using namespace cv;

static QImage toQImage(const Mat &mat)
{
    // Convert to 8U depth
    Mat mat8u;
    if (mat.depth() != CV_8U) {
        double globalMin = std::numeric_limits<double>::max();
        double globalMax = -std::numeric_limits<double>::max();

        std::vector<Mat> mv;
        split(mat, mv);
        for (size_t i=0; i<mv.size(); i++) {
            double min, max;
            minMaxLoc(mv[i], &min, &max);
            globalMin = std::min(globalMin, min);
            globalMax = std::max(globalMax, max);
        }
        assert(globalMax >= globalMin);

        double range = globalMax - globalMin;
        if (range != 0) {
            double scale = 255 / range;
            convertScaleAbs(mat, mat8u, scale, -(globalMin * scale));
        } else {
            // Monochromatic
            mat8u = Mat(mat.size(), CV_8UC1, Scalar((globalMin+globalMax)/2));
        }
    } else {
        mat8u = mat;
    }

    // Convert to 3 channels
    Mat mat8uc3;
    if      (mat8u.channels() == 4) cvtColor(mat8u, mat8uc3, CV_BGRA2RGB);
    else if (mat8u.channels() == 3) cvtColor(mat8u, mat8uc3, CV_BGR2RGB);
    else if (mat8u.channels() == 1) cvtColor(mat8u, mat8uc3, CV_GRAY2RGB);

    return QImage(mat8uc3.data, mat8uc3.cols, mat8uc3.rows, 3*mat8uc3.cols, QImage::Format_RGB888).copy();
}

FaceViewer::FaceViewer(QWidget *parent)
    : TemplateViewer(parent)
{
    setAcceptDrops(true);
    setFile(File());
    selectionDistance = 4.0;
    editable = false;
    dragging = false;
    update();
}

void FaceViewer::setFile(const File &file_)
{
    this->file = file_;

    // Update landmarks
    landmarks.clear();
    if (file.contains("Affine_0")) landmarks.append(file.get<QPointF>("Affine_0"));
    if (file.contains("Affine_1")) landmarks.append(file.get<QPointF>("Affine_1"));
    while (landmarks.size() < numLandmarks)
        landmarks.append(QPointF());
    nearestLandmark = -1;

    QtConcurrent::run(this, &FaceViewer::refreshImage);
}

void FaceViewer::refreshImage()
{
    if (file.isNull()) {
        setImage(file, true);
    } else {
        Transform *t = Transform::make("Open", NULL);

        Template dst;
        t->project(file, dst);

        QImage image = toQImage(dst.m());
        setImage(image, true);

        double rows = dst.m().rows;
        double cols = dst.m().cols;

        // Update selection distance to be proportial to the largest image dimension
        selectionDistance = rows > cols ? rows*proportionality : cols*proportionality;

        delete t;
    }
}

QRect FaceViewer::getImageRect(const QPointF &ip, const QSize &size) const
{
    if (!pixmap() || isNull()) return QRect();

    QRect ir = QRect(ip.x() - size.width()/2.0,
                         ip.y() - size.height()/2.0, size.width(), size.height());

    if ((ir.x() < 0) || (ir.x() > imageWidth()) || (ir.y() < 0) || (ir.y() > imageHeight())) return QRect();

    return ir;
}

QRectF FaceViewer::getScreenRect(const QPointF &sp, int width_, int height_) const
{
    if (!pixmap() || isNull()) return QRectF();

    QRectF sr = QRectF(sp.x() - width_/2.0, sp.y() - height_/2.0, width_, height_);

    return sr;
}

void FaceViewer::mousePressEvent(QMouseEvent *event)
{
    ImageViewer::mousePressEvent(event);
    event->accept();

    if (isNull()) return;

    if (!editable) {
        //emit selectedInput(index);
        return;
    }

    if (event->button() == Qt::LeftButton && nearestLandmark != -1) dragging = true;

    update();
}

void FaceViewer::mouseMoveEvent(QMouseEvent *event)
{
    ImageViewer::mouseMoveEvent(event);
    event->accept();

    mousePoint = getImagePoint(event->pos());

    if (dragging) {
        landmarks[nearestLandmark] = getImagePoint(event->pos());
    } else {
        nearestLandmark = -1;
        if (!mousePoint.isNull()) {
            double nearestDistance = std::numeric_limits<double>::max();
            for (int i=0; i<numLandmarks; i++) {
                if (landmarks[i].isNull()) {
                    nearestLandmark = -1;
                    break;
                }

                double dist = sqrt(pow(mousePoint.x() - landmarks[i].x(), 2.0) + pow(mousePoint.y() - landmarks[i].y(), 2.0));

                if (dist < nearestDistance && dist < selectionDistance) {
                    nearestDistance = dist;
                    nearestLandmark = i;
                }
            }
        } else {
            unsetCursor();
        }
    }

    update();
}

void FaceViewer::mouseReleaseEvent(QMouseEvent *event)
{
    ImageViewer::mouseReleaseEvent(event);
    event->accept();

    if (dragging) {
        dragging = false;
        emit newLandmarks(QStringList() << "Affine_0" << "Affine_1", landmarks);
    }

    update();
}

void FaceViewer::paintEvent(QPaintEvent *event)
{
    static const QColor nearest(0, 255, 0, 192);
    static const QColor normal(255, 0, 0, 192);

    ImageViewer::paintEvent(event);
    event->accept();

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    for (int i=0; i<numLandmarks; i++) {
        if (!landmarks[i].isNull() && editable) {
            if (i == nearestLandmark) painter.setBrush(QBrush(nearest));
            else painter.setBrush(QBrush(normal));
            painter.drawEllipse(getScreenPoint(landmarks[i]), 4, 4);
        }
    }

    if (!isNull() && dragging && editable) {
        setCursor(QCursor(Qt::BlankCursor));

        QSize reticleSize = src.size() *.1;
        QSize displaySize = pixmap()->size() *.3;

        QRect reticle = getImageRect(mousePoint, reticleSize);
        QImage reticleImage = src.copy(reticle).scaled(src.size()*2.0, Qt::KeepAspectRatio);

        QRectF displayRect = getScreenRect(getScreenPoint(mousePoint), displaySize.width(), displaySize.height());
        painter.drawImage(displayRect, reticleImage);

        painter.setPen(QPen(nearest));
        painter.drawLine(getScreenPoint(QPointF(mousePoint.x(), mousePoint.y() - reticleSize.height())), getScreenPoint(QPointF(mousePoint.x(), mousePoint.y() + reticleSize.height())));
        painter.drawLine(getScreenPoint(QPointF(mousePoint.x() - reticleSize.width(), mousePoint.y())), getScreenPoint(QPointF(mousePoint.x() + reticleSize.width(), mousePoint.y())));
    }
    else setCursor(QCursor(Qt::ArrowCursor));
}
