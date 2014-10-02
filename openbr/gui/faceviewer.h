#ifndef BR_FACEVIEWER_H
#define BR_FACEVIEWER_H

#include <openbr/openbr_plugin.h>
#include <openbr/gui/templateviewer.h>

namespace br
{

class BR_EXPORT FaceViewer : public TemplateViewer
{
    Q_OBJECT

    int index;
    double selectionDistance;
    bool dragging;

public:
    explicit FaceViewer(QWidget *parent = 0);

public slots:
    void setFile(const br::File &file_);

protected slots:
    void mouseMoveEvent(QMouseEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void paintEvent(QPaintEvent *event);

private:
    void refreshImage();
    QRect getImageRect(const QPointF &ip, const QSize &size) const;
    QRectF getScreenRect(const QPointF &sp, int width_, int height_) const;

signals:
    void newLandmarks(QStringList, QList<QPointF>);
};

}

#endif // BR_FACEVIEWER_H
