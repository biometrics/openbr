#ifndef BR_TAIL_H
#define BR_TAIL_H

#include <QBoxLayout>
#include <QFutureWatcher>
#include <QKeyEvent>
#include <QLabel>
#include <QSlider>
#include <QString>
#include <QWheelEvent>
#include <openbr/openbr_plugin.h>

namespace br
{

class BR_EXPORT Tail : public QWidget
{
    Q_OBJECT
    QHBoxLayout *layout;
    QSlider *slider;
    QLabel *lhs, *rhs;
    File targetGallery, queryGallery;
    FileList targetFiles, queryFiles;
    QList<float> scores;
    QFutureWatcher<void> compareWatcher;

public:
    explicit Tail(QWidget *parent = 0);

public slots:
    void clear();
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
    void compareDone();

signals:
    void newTargetFile(File file);
    void newQueryFile(File file);
    void newScore(float score);
};

} // namespace br

#endif // BR_TAIL_H
