#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup galleries
 * \brief Reads/writes OpenCV's .vec format.
 * \author Scott Klum \cite sklum
 */

class vecGallery : public Gallery
{
    Q_OBJECT

    Q_PROPERTY(int width READ get_width WRITE set_width RESET reset_width STORED false)
    Q_PROPERTY(int height READ get_height WRITE set_height RESET reset_height STORED false)
    BR_PROPERTY(int, width, 24)
    BR_PROPERTY(int, height, 24)

    TemplateList readBlock(bool *done)
    {
        *done = true;

        QFile gallery;
        gallery.setFileName(file);
        if (!gallery.exists())
            qFatal("File %s does not exist", qPrintable(gallery.fileName()));

        QFile::OpenMode mode = QFile::ReadOnly;
        if (!gallery.open(mode))
            qFatal("Can't open gallery: %s for reading", qPrintable(gallery.fileName()));

        // Read header
        int count, size;
        short temp;

        const size_t read1 = gallery.read((char*)&count,sizeof(count));
        const size_t read2 = gallery.read((char*)&size,sizeof(size));
        const size_t read3 = gallery.read((char*)&temp,sizeof(temp));
        const size_t read4 = gallery.read((char*)&temp,sizeof(temp));

        if (read1 != sizeof(count) || read2 != sizeof(size) || read3 != sizeof(temp) || read4 != sizeof(temp))
            qFatal("Failed to read header.");

        if (size != width*height)
            qFatal("Width*height != vector size.");

        // Read content
        short *vec = new short[size];

        TemplateList templates;
        for (int i=0; i<count; i++) {
            uchar tmp = 0;
            const size_t read5 = gallery.read((char*)&tmp,sizeof(tmp));
            const size_t read6 = gallery.read((char*)vec,size*sizeof(short));

            if (read5 != sizeof(tmp) || read6 != size*sizeof(short))
                qFatal("Unable to read vector.");

            cv::Mat m(height, width, CV_8UC1);
            for (int r = 0; r < height; r++)
                for (int c = 0; c < width; c++)
                    m.ptr(r)[c] = (uchar)vec[r*width+c];
	    Template t(m);
	    t.file.set("Label",1);
            templates.append(t);
        }

        return templates;
    }

    void write(const Template &t)
    {
	Q_UNUSED(t);
    }
};

BR_REGISTER(Gallery, vecGallery)

} // namespace br

#include "gallery/vec.moc"

