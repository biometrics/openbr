#include "galleryviewer.h"

using namespace br;

/**** GALLERY_VIEWER ****/
/*** PUBLIC ***/
GalleryViewer::GalleryViewer(QWidget *parent)
    : QMainWindow(parent)
{
    addToolBar(Qt::TopToolBarArea, &gGallery);
    setCentralWidget(&tvgTemplateViewerGrid);
    setWindowFlags(Qt::Widget);

    connect(&tvgTemplateViewerGrid, SIGNAL(newInput(br::File)), &gGallery, SLOT(enroll(br::File)));
    connect(&tvgTemplateViewerGrid, SIGNAL(newInput(QImage)), &gGallery, SLOT(enroll(QImage)));
    connect(&tvgTemplateViewerGrid, SIGNAL(selectedInput(br::File)), &gGallery, SLOT(select(br::File)));

    tmTemplateMetadata.addClassifier("GenderClassification", "MM0");
    tmTemplateMetadata.addClassifier("AgeRegression", "MM0");
    setFiles(FileList());
}

/*** PUBLIC SLOTS ***/
void GalleryViewer::setAlgorithm(const QString &algorithm)
{
    tmTemplateMetadata.setAlgorithm(algorithm);
    setFiles(FileList());
}

void GalleryViewer::setFiles(FileList files)
{
    bool same = true;
    foreach (const File &file, files) {
        if (file != files.first()) {
            same = false;
            break;
        }
    }
    if (same) files = files.mid(0, 1);

    tvgTemplateViewerGrid.setFiles(files);
    tmTemplateMetadata.setVisible(files.size() == 1);
    if (files.size() == 1)
        tmTemplateMetadata.setFile(files.first());
}

#include "moc_galleryviewer.cpp"
