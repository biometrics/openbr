#ifndef BR_GALLERYTOOLBAR_H
#define BR_GALLERYTOOLBAR_H

#include <QFutureWatcher>
#include <QImage>
#include <QLabel>
#include <QMutex>
#include <QString>
#include <QTimer>
#include <QToolBar>
#include <QToolButton>
#include <QWidget>
#include <openbr/openbr_plugin.h>

namespace br
{

class BR_EXPORT GalleryToolBar : public QToolBar
{
    Q_OBJECT
    br::File input, gallery;
    br::FileList files;

    QLabel lGallery;
    QToolButton tbOpenFile, tbOpenFolder, tbWebcam, tbBack, tbMean;
    QFutureWatcher<void> enrollmentWatcher;
    QTimer timer;

    static QMutex galleryLock;

public:
    explicit GalleryToolBar(QWidget *parent = 0);

public slots:
    void enroll(const br::File &input);
    void enroll(const QImage &input);
    void select(const br::File &file);

private:
    void _enroll(const br::File &input);
    void _checkWebcam();

private slots:
    void checkWebcam();
    void enrollmentFinished();
    void home();
    void mean();
    void openFile();
    void openFolder();

signals:
    void newGallery(br::File gallery);
    void newFiles(br::FileList files);
};

} // namespace br

#endif // BR_GALLERYTOOLBAR_H
