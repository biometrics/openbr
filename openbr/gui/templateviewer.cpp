#include <QColor>
#include <QMimeData>
#include <QPainter>
#include <QPen>
#include <QUrl>
#include <QtConcurrentRun>
#include <openbr/openbr.h>

#include "templateviewer.h"

using namespace br;

/*** STATIC ***/
const int NumLandmarks = 2;

static bool lessThan(const QPointF &a, const QPointF &b)
{
    return ((a.x() < b.x()) ||
            ((a.x() == b.x()) && (a.y() < b.y())));
}

/*** PUBLIC ***/
TemplateViewer::TemplateViewer(QWidget *parent)
    : ImageViewer(parent)
{
    setAcceptDrops(true);
    setMouseTracking(true);
    setDefaultText("<b>Drag Photo or Folder Here</b>\n");
    format = "Photo";
    editable = false;
    setFile(File());
    update();
}

/*** PUBLIC SLOTS ***/
void TemplateViewer::setFile(const File &file_)
{
    this->file = file_;

    // Update landmarks
    landmarks.clear();
    if (file.contains("Affine_0")) landmarks.append(file.get<QPointF>("Affine_0"));
    if (file.contains("Affine_1")) landmarks.append(file.get<QPointF>("Affine_1"));
    while (landmarks.size() < NumLandmarks)
        landmarks.append(QPointF());
    nearestLandmark = -1;

    TemplateViewer::refreshImage();
}

void TemplateViewer::setEditable(bool enabled)
{
    editable = enabled;
    update();
}

void TemplateViewer::setMousePoint(const QPointF &mousePoint)
{
    this->mousePoint = mousePoint;
    update();
}

void TemplateViewer::setFormat(const QString &format)
{
    this->format = format;
    QtConcurrent::run(this, &TemplateViewer::refreshImage);
}

/*** PRIVATE ***/
void TemplateViewer::refreshImage()
{
    if (file.isNull() || (format == "Photo")) {
        setImage(file, true);
    } else {
        const QString path = QString(br::Globals->scratchPath()) + "/thumbnails";
        const QString hash = file.hash()+format;
        const QString processedFile = path+"/"+file.baseName()+hash+".png";
        if (!QFileInfo(processedFile).exists()) {
            if (format == "Registered")
                Enroll(file.flat(), path+"[postfix="+hash+",cache,algorithm=RegisterAffine]");
            else if (format == "Enhanced")
                Enroll(file.flat(), path+"[postfix="+hash+",cache,algorithm=ContrastEnhanced]");
            else if (format == "Features")
                Enroll(file.flat(), path+"[postfix="+hash+",cache,algorithm=ColoredLBP]");
        }
        setImage(processedFile, true);
    }
}

QPointF TemplateViewer::getImagePoint(const QPointF &sp) const
{
    if (!pixmap() || isNull()) return QPointF();
    QPointF ip = QPointF(imageWidth()*(sp.x() - (width() - pixmap()->width())/2)/pixmap()->width(),
                        imageHeight()*(sp.y() - (height() - pixmap()->height())/2)/pixmap()->height());
    if ((ip.x() < 0) || (ip.x() > imageWidth()) || (ip.y() < 0) || (ip.y() > imageHeight())) return QPointF();
    return ip;
}

QPointF TemplateViewer::getScreenPoint(const QPointF &ip) const
{
    if (!pixmap() || isNull()) return QPointF();
    return QPointF(ip.x()*pixmap()->width()/imageWidth() + (width()-pixmap()->width())/2,
                   ip.y()*pixmap()->height()/imageHeight() + (height()-pixmap()->height())/2);
}

/*** PROTECTED SLOTS ***/
void TemplateViewer::dragEnterEvent(QDragEnterEvent *event)
{
    ImageViewer::dragEnterEvent(event);
    event->accept();

    if (event->mimeData()->hasUrls() || event->mimeData()->hasImage())
        event->acceptProposedAction();
}

void TemplateViewer::dropEvent(QDropEvent *event)
{
    ImageViewer::dropEvent(event);
    event->accept();    

    event->acceptProposedAction();
    const QMimeData *mimeData = event->mimeData();
    if (mimeData->hasImage()) {
        QImage input = qvariant_cast<QImage>(mimeData->imageData());
        emit newInput(input);
    } else if (event->mimeData()->hasUrls()) {
        File input;
        foreach (const QUrl &url, mimeData->urls()) {
            if (!url.isValid()) continue;
            if (url.toString().startsWith("http://images.google.com/search")) continue; // Not a true image URL
            const QString localFile = url.toLocalFile();
            if (localFile.isNull()) input.append(url.toString());
            else                    input.append(localFile);
        }
        if (input.isNull()) return;
        emit newInput(input);
    }
}

void TemplateViewer::leaveEvent(QEvent *event)
{
    ImageViewer::leaveEvent(event);
    event->accept();
    clearFocus();

    nearestLandmark = -1;
    mousePoint = QPointF();
    emit newMousePoint(mousePoint);
    unsetCursor();
    update();
}

void TemplateViewer::mouseMoveEvent(QMouseEvent *event)
{
    ImageViewer::mouseMoveEvent(event);
    event->accept();

    mousePoint = getImagePoint(event->pos());
    nearestLandmark = -1;
    if (!mousePoint.isNull()) {
        double nearestDistance = std::numeric_limits<double>::max();
        for (int i=0; i<NumLandmarks; i++) {
            if (landmarks[i].isNull()) {
                nearestLandmark = -1;
                break;
            }

            double dist = sqrt(pow(mousePoint.x() - landmarks[i].x(), 2.0) + pow(mousePoint.y() - landmarks[i].y(), 2.0));
            if (dist < nearestDistance) {
                nearestDistance = dist;
                nearestLandmark = i;
            }
        }

        if (format == "Photo") unsetCursor();
        else                   setCursor(Qt::BlankCursor);
    } else {
        unsetCursor();
    }

    emit newMousePoint(mousePoint);
    update();
}

void TemplateViewer::mousePressEvent(QMouseEvent *event)
{
    ImageViewer::mousePressEvent(event);
    event->accept();

    if (isNull() || getImagePoint(event->pos()).isNull()) return;

    if (!editable || (format != "Photo")) {
        emit selectedInput(file);
        return;
    }

    int index;
    for (index=0; index<NumLandmarks; index++)
        if (landmarks[index].isNull()) break;

    if ((event->button() == Qt::RightButton) || (index == NumLandmarks)) {
        // Remove nearest point
        if (nearestLandmark == -1) return;
        index = nearestLandmark;
        landmarks[nearestLandmark] = QPointF();
        nearestLandmark = -1;
    }

    if (event->button() == Qt::LeftButton) {
        // Add a point
        landmarks[index] = getImagePoint(event->pos());
        qSort(landmarks.begin(), landmarks.end(), lessThan);
        if (!landmarks.contains(QPointF()))
            emit selectedInput(file.name+QString("[Affine_0_X=%1, Affine_0_Y=%2, Affine_1_X=%3, Affine_1_Y=%4]").arg(QString::number(landmarks[0].x()),
                                                                                                                                      QString::number(landmarks[0].y()),
                                                                                                                                      QString::number(landmarks[1].x()),
                                                                                                                                      QString::number(landmarks[1].y())));
    }

    update();
}

void TemplateViewer::paintEvent(QPaintEvent *event)
{
    static const QColor nearest(0, 0, 255, 192);
    static const QColor normal(0, 255, 0, 192);

    ImageViewer::paintEvent(event);
    event->accept();

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    if (format == "Photo") {
        for (int i=0; i<NumLandmarks; i++) {
            if (!landmarks[i].isNull() && editable) {
                if (i == nearestLandmark) painter.setBrush(QBrush(nearest));
                else                  painter.setBrush(QBrush(normal));
                painter.drawEllipse(getScreenPoint(landmarks[i]), 4, 4);
            }
        }
    } else {
        if (!mousePoint.isNull() && !isNull()) {
            painter.setPen(QPen(normal));
            painter.drawLine(getScreenPoint(QPointF(mousePoint.x(), 0)), getScreenPoint(QPointF(mousePoint.x(), imageHeight())));
            painter.drawLine(getScreenPoint(QPointF(0, mousePoint.y())), getScreenPoint(QPointF(imageWidth(), mousePoint.y())));
        }
    }
}

#include "moc_templateviewer.cpp"
