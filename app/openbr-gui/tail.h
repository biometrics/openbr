#ifndef TAIL_H
#define TAIL_H

#include <QLabel>
#include <QKeyEvent>
#include <QMainWindow>
#include <QString>
#include <QToolBar>
#include <QToolButton>
#include <QWheelEvent>
#include <QWidget>
#include <openbr_plugin.h>

#include "dataset.h"
#include "score.h"
#include "scoredistribution.h"

namespace br
{

class BR_EXPORT_GUI Tail : public QMainWindow
{
    Q_OBJECT
    QToolBar tbToolBar;
    QWidget wLeftSpacer, wRightSpacer;
    QToolButton tbFirst, tbPrevious, tbNext, tbLast;
    Dataset dDataset;
    ScoreDistribution sdScoreDistribution;
    Score sScore;

    br::File target, query;
    bool targetLocked, queryLocked;

    int count;
    int index;
    br::FileList targets, queries;
    QList<float> scores;

public:
    explicit Tail(QWidget *parent = 0);

public slots:
    void setAlgorithm(const QString &algorithm);
    void setIndex(int index);
    void setTargetGallery(const br::File &gallery);
    void setQueryGallery(const br::File &gallery);
    void setTargetGalleryFiles(const br::FileList &files);
    void setQueryGalleryFiles(const br::FileList &files);
    void setCount(int count);
    void setThreshold(float score);

protected:
    void keyPressEvent(QKeyEvent *event);
    void wheelEvent(QWheelEvent *event);

private:
    void compare();

private slots:
    void updateInterface();
    void import(QString tailFile = "");
    void first();
    void previous();
    void next();
    void last();
    void lock(bool checked);
    void selected(QPointF point);

signals:
    void newTargetFiles(br::FileList files);
    void newQueryFiles(br::FileList files);
};

} // namespace br

#endif // TAIL_H
