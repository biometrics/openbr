#include <QDateTime>
#include <QDir>
#include <QFileDialog>
#include <QIcon>
#include <QMessageBox>
#include <QSharedPointer>
#include <QSize>
#include <QStandardPaths>
#include <QtConcurrentRun>
#include <opencv2/highgui/highgui.hpp>
#include <assert.h>
#include <openbr/openbr_plugin.h>

#include "gallerytoolbar.h"
#include "utility.h"

/**** GALLERY ****/
/*** STATIC ***/
QMutex br::GalleryToolBar::galleryLock;

/*** PUBLIC ***/
br::GalleryToolBar::GalleryToolBar(QWidget *parent)
    : QToolBar("Gallery", parent)
{
    lGallery.setAlignment(Qt::AlignCenter);
    lGallery.setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    tbOpenFile.setIcon(QIcon(":/glyphicons/png/glyphicons_138_picture@2x.png"));
    tbOpenFile.setToolTip("Load Photo");
    tbOpenFolder.setIcon(QIcon(":/glyphicons/png/glyphicons_144_folder_open@2x.png"));
    tbOpenFolder.setToolTip("Load Photo Folder");
    tbWebcam.setCheckable(true);
    tbWebcam.setIcon(QIcon(":/glyphicons/png/glyphicons_301_webcam@2x.png"));
    tbWebcam.setToolTip("Load Webcam");
    tbBack.setEnabled(false);
    tbBack.setIcon(QIcon(":/glyphicons/png/glyphicons_221_unshare@2x.png"));
    tbBack.setToolTip("Back");
    tbMean.setIcon(QIcon(":/glyphicons/png/glyphicons_003_user@2x.png"));
    tbMean.setToolTip("Mean Image");

    addWidget(&tbOpenFile);
    addWidget(&tbOpenFolder);
    addWidget(&tbWebcam);
    addSeparator();
    addWidget(&lGallery);
    addSeparator();
    addWidget(&tbBack);
    addWidget(&tbMean);
    setIconSize(QSize(20,20));

    connect(&tbOpenFile, SIGNAL(clicked()), this, SLOT(openFile()));
    connect(&tbOpenFolder, SIGNAL(clicked()), this, SLOT(openFolder()));
    connect(&tbBack, SIGNAL(clicked()), this, SLOT(home()));
    connect(&tbMean, SIGNAL(clicked()), this, SLOT(mean()));
    connect(&enrollmentWatcher, SIGNAL(finished()), this, SLOT(enrollmentFinished()));
    connect(&timer, SIGNAL(timeout()), this, SLOT(checkWebcam()));

    timer.start(500);
}

/*** PUBLIC SLOTS ***/
void br::GalleryToolBar::enroll(const br::File &input)
{
    if (input.isNull()) return;
    enrollmentWatcher.setFuture(QtConcurrent::run(this, &GalleryToolBar::_enroll, input));
}

void br::GalleryToolBar::enroll(const QImage &input)
{
    QString tempFileName = br::Context::scratchPath() + "/tmp/" + QDateTime::currentDateTime().toString("yyyy-MM-ddThh:mm:ss:zzz") + ".png";
    input.save(tempFileName);
    enroll(tempFileName);
}

void br::GalleryToolBar::select(const br::File &file)
{
    tbBack.setEnabled(true);
    emit newFiles(br::FileList() << file);
    emit newGallery(file);
}

/*** PRIVATE ***/
void br::GalleryToolBar::_enroll(const br::File &input)
{
    galleryLock.lock();
    this->input = input;
    gallery = input.name + ".mem";
    br::Enroll(input.flat(), gallery.flat());
    files = FileList::fromGallery(gallery);

    galleryLock.unlock();
}

void br::GalleryToolBar::_checkWebcam()
{
    static QSharedPointer<cv::VideoCapture> videoCapture;
    if (videoCapture.isNull()) {
        videoCapture = QSharedPointer<cv::VideoCapture>(new cv::VideoCapture(0));
        cv::Mat m;
        while (!m.data) videoCapture->read(m); // First frames can be empty
    }

    if (galleryLock.tryLock()) {
        cv::Mat m;
        videoCapture->read(m);
        galleryLock.unlock();
        enroll(toQImage(m));
    }
}

/*** PRIVATE SLOTS ***/
void br::GalleryToolBar::checkWebcam()
{
    // Check webcam
    if (!tbWebcam.isChecked()) return;
    QtConcurrent::run(this, &GalleryToolBar::_checkWebcam);
}

void br::GalleryToolBar::enrollmentFinished()
{
    if (files.isEmpty()) {
        if (input.get<bool>("enrollAll", false) && !tbWebcam.isChecked()) {
            QMessageBox msgBox;
            msgBox.setText("Quality test failed.");
            msgBox.setInformativeText("Enroll anyway?");
            msgBox.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
            msgBox.setDefaultButton(QMessageBox::Ok);
            int ret = msgBox.exec();

            if (ret == QMessageBox::Ok) {
                br::File file = input;
                file.set("enrollAll", QVariant(false));
                enroll(file);
            }
        }
        return;
    }

    lGallery.setText(input.baseName() + (files.size() != 1 ? " (" + QString::number(files.size()) + ")" : ""));
    tbBack.setEnabled(false);
    emit newFiles(files);
    emit newGallery(gallery);
}

void br::GalleryToolBar::home()
{
    tbBack.setEnabled(false);
    emit newGallery(gallery);
}

void br::GalleryToolBar::mean()
{
    const QString file = QString("%1/mean/%2.png").arg(br::Globals->scratchPath(), input.baseName()+input.hash());
    br_set_property("CENTER_TRAIN_B", qPrintable(file));
    br::File trainingFile = input;
    br_train(qPrintable(trainingFile.flat()), "[algorithm=MedianFace]");
    enroll(file);
}

void br::GalleryToolBar::openFile()
{
    enroll(QFileDialog::getOpenFileName(this, "Select Photo", QStandardPaths::standardLocations(QStandardPaths::PicturesLocation).first()));
}

void br::GalleryToolBar::openFolder()
{
    enroll(QFileDialog::getExistingDirectory(this, "Select Photo Directory", QStandardPaths::standardLocations(QStandardPaths::HomeLocation).first()));
}

#include "moc_gallerytoolbar.cpp"
