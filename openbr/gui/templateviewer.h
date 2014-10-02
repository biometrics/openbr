#ifndef BR_TEMPLATEVIEWER_H
#define BR_TEMPLATEVIEWER_H

#include <QDragEnterEvent>
#include <QDropEvent>
#include <QEvent>
#include <QImage>
#include <QList>
#include <QMouseEvent>
#include <QPointF>
#include <QString>
#include <QWidget>
#include <openbr/openbr_plugin.h>
#include <openbr/gui/imageviewer.h>

namespace br
{

class BR_EXPORT TemplateViewer : public ImageViewer
{
    Q_OBJECT

public:
    explicit TemplateViewer(QWidget *parent = 0);

public slots:
    void setFile(const br::File &file);
    void setEditable(bool enabled);
    void setMousePoint(const QPointF &mousePoint);
    void setFormat(const QString &format);

protected:
    File file;
    QPointF mousePoint;
    QString format;

    bool editable;
    QList<QPointF> landmarks;
    int nearestLandmark;
    QPointF getImagePoint(const QPointF &sp) const;
    QPointF getScreenPoint(const QPointF &ip) const;

private:
    void refreshImage();

protected slots:
    void dragEnterEvent(QDragEnterEvent *event);
    void dropEvent(QDropEvent *event);
    void leaveEvent(QEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void paintEvent(QPaintEvent *event);

signals:
    void newInput(br::File);
    void newInput(QImage);
    void newMousePoint(QPointF);
    void selectedInput(br::File);
};

}

#endif // BR_TEMPLATEVIEWER_H
