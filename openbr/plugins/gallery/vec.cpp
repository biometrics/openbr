#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup galleries
 * \brief Reads/writes OpenCV's .vec format.
 * \author Scott Klum \cite sklum
 */

class vecGallery : public FileGallery
{
    Q_OBJECT

    Q_PROPERTY(int width READ get_width WRITE set_width RESET reset_width STORED false)
    Q_PROPERTY(int height READ get_height WRITE set_height RESET reset_height STORED false)
    BR_PROPERTY(int, width, 24)
    BR_PROPERTY(int, height, 24)

    QList<cv::Mat> mats;

    ~vecGallery()
    {
        if (mats.isEmpty())
            return;

        writeOpen();

        // Write header
        int count = mats.size();
        int size = width*height;
        short temp = 0;

        const size_t write1 = f.write((char*)&count,sizeof(count));
        const size_t write2 = f.write((char*)&size,sizeof(size));
        const size_t write3 = f.write((char*)&temp,sizeof(temp));
        const size_t write4 = f.write((char*)&temp,sizeof(temp));

        if (write1 != sizeof(count) || write2 != sizeof(size) || write3 != sizeof(temp) || write4 != sizeof(temp))
            qFatal("Failed to write header.");

        for (int i=0; i<count; i++) {
            uchar tmp = 0;
            const size_t write5 = f.write((char*)&tmp,sizeof(tmp));
            Q_UNUSED(write5);
            for (int r = 0; r < height; r++)
                for (int c = 0; c < width; c++) {
                    short buffer = mats[i].ptr(r)[c];
                    f.write((char*)&buffer, sizeof(buffer));
                }
        }

        f.close();
    }

    TemplateList readBlock(bool *done)
    {
        readOpen();

        *done = true;

        // Read header
        int count, size;
        short temp;

        const size_t read1 = f.read((char*)&count,sizeof(count));
        const size_t read2 = f.read((char*)&size,sizeof(size));
        const size_t read3 = f.read((char*)&temp,sizeof(temp));
        const size_t read4 = f.read((char*)&temp,sizeof(temp));

        if (read1 != sizeof(count) || read2 != sizeof(size) || read3 != sizeof(temp) || read4 != sizeof(temp))
            qFatal("Failed to read header.");

        if (size != width*height)
            qFatal("Width*height != vector size.");

        // Read content
        short *vec = new short[size];

        TemplateList templates;
        for (int i=0; i<count; i++) {
            uchar tmp = 0;
            const size_t read5 = f.read((char*)&tmp,sizeof(tmp));
            const size_t read6 = f.read((char*)vec,size*sizeof(short));

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
        if (t.m().rows == height && t.m().cols == width && t.m().type() == CV_8UC1)
            mats.append(t);
        else
            qFatal("Matrix has incorrect width/height/type.");
    }
};

BR_REGISTER(Gallery, vecGallery)

} // namespace br

#include "gallery/vec.moc"

