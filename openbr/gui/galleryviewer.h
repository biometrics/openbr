#ifndef BR_GALLERYVIEWER_H
#define BR_GALLERYVIEWER_H

#include <QMainWindow>
#include <QWidget>
#include <openbr/openbr_plugin.h>

#include "gallerytoolbar.h"
#include "templateviewergrid.h"
#include "templatemetadata.h"

namespace br
{

class BR_EXPORT GalleryViewer : public QMainWindow
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

#endif // BR_GALLERYVIEWER_H
