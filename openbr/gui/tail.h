#ifndef BR_TAIL_H
#define BR_TAIL_H

#include <QKeyEvent>
#include <QSlider>
#include <QString>
#include <QWheelEvent>
#include <openbr/openbr_plugin.h>

namespace br
{

class BR_EXPORT Tail : public QSlider
{
    Q_OBJECT
    int count;
    File target, query;
    FileList targets, queries;
    QList<float> scores;

public:
    explicit Tail(QWidget *parent = 0);

public slots:
    void setIndex(int index);
    void setTargetGallery(const File &gallery);
    void setQueryGallery(const File &gallery);
    void setTargetGalleryFiles(const FileList &files);
    void setQueryGalleryFiles(const FileList &files);
    void setCount(int count);
    void setThreshold(float score);

protected:
    void keyPressEvent(QKeyEvent *event);
    void wheelEvent(QWheelEvent *event);

private:
    void compare();

private slots:
    void import(QString tailFile = "");
    void first();
    void previous();
    void next();
    void last();
    void selected(QPointF point);

signals:
    void newTargetFile(File file);
    void newTargetFiles(FileList files);
    void newQueryFile(File file);
    void newQueryFiles(FileList files);
};

} // namespace br

#endif // BR_TAIL_H
