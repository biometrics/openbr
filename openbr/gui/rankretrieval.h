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

    File target, query;
    FileList targetFiles, queryFiles;
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

    void compareDone();

signals:
    void newTargetFileList(FileList file);
    void newQueryFile(File file);
    void newScore(float score);

private:
    void enroll();
    void compare();

};

} // namespace br

#endif // BR_RANKRETRIEVAL_H
