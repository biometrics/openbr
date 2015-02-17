#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/qtutils.h>

namespace br
{

void FileGallery::init()
{
    f.setFileName(file);

    Gallery::init();
}

void FileGallery::writeOpen()
{
    if (!f.isOpen() ) {
        QtUtils::touchDir(f);
        if (!f.open(QFile::WriteOnly))
            qFatal("Failed to open %s for writing.", qPrintable(file));
    }
}

bool FileGallery::readOpen()
{
    if (!f.isOpen() ) {
        if (!f.exists() ) {
            qFatal("File %s does not exist.", qPrintable(file));
        }

        if (!f.open(QFile::ReadOnly))
            qFatal("Failed to open %s for reading.", qPrintable(file));
        return true;
    }
    return false;
}

qint64 FileGallery::totalSize()
{
    readOpen();
    return f.size();
}

} // namespace br
