#ifndef GALLERYVIEWER_H
#define GALLERYVIEWER_H

#include <QMainWindow>
#include <QWidget>
#include <openbr_plugin.h>

#include "gallerytoolbar.h"
#include "templateviewergrid.h"
#include "templatemetadata.h"

namespace br
{

class BR_EXPORT_GUI GalleryViewer : public QMainWindow
{
    Q_OBJECT

public:
    GalleryToolBar gGallery;
    TemplateViewerGrid tvgTemplateViewerGrid;
    TemplateMetadata tmTemplateMetadata;

    explicit GalleryViewer(QWidget *parent = 0);

public slots:
    void setAlgorithm(const QString &algorithm);
    void setFiles(br::FileList files);
};

} // namespace br

#endif // GALLERYVIEWER_H
