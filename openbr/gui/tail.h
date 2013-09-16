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
    File targetGallery, queryGallery;
    FileList targetFiles, queryFiles;
    QList<float> scores;

public:
    explicit Tail(QWidget *parent = 0);

public slots:
    void setIndex(int index);
    void setTargetGallery(const File &gallery);
    void setQueryGallery(const File &gallery);

protected:
    void keyPressEvent(QKeyEvent *event);
    void wheelEvent(QWheelEvent *event);

private:
    void compare();

private slots:
    void first();
    void previous();
    void next();
    void last();

signals:
    void newTargetFile(File file);
    void newQueryFile(File file);
    void newScore(float score);
};

} // namespace br

#endif // BR_TAIL_H
