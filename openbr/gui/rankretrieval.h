#ifndef BR_RANKRETRIEVAL_H
#define BR_RANKRETRIEVAL_H

#include <QFutureWatcher>
#include <QFileDialog>
#include <openbr/openbr_plugin.h>
#include <openbr/gui/faceviewer.h>

namespace br {

class BR_EXPORT RankRetrieval : public QWidget
{
    Q_OBJECT

    int gridPage, gridSize;
    File target, query;
    FileList matches;
    QList<float> scores;
    QFutureWatcher<void> enrollWatcher;
    QFutureWatcher<void> compareWatcher;

    QString targetPath;

public:
    explicit RankRetrieval(QWidget *parent = 0);

public slots:
    void setAlgorithm(const QString &algorithm);

    void clear();
    void setIndex(int index);
    void setTargetGallery(const File &file);
    void setTargetPath();
    void setQueryGallery(const File &file);

    void first();
    void previous();
    void next();
    void last();

    void setGridSize(const QString &size);

    void compare();

private slots:
    void compareDone();

signals:
    void newTargetFileList(FileList);
    void newQueryFile(File);
    void newScore(float);

private:
    void enroll();
    void display();

};

} // namespace br

#endif // BR_RANKRETRIEVAL_H
