#include <openbr/plugins/openbr_internal.h>

namespace br
{

class GalleryOutputTransform : public TimeVaryingTransform
{
    Q_OBJECT

    Q_PROPERTY(QString outputString READ get_outputString WRITE set_outputString RESET reset_outputString STORED false)
    BR_PROPERTY(QString, outputString, "")

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        if (src.empty())
            return;
        dst = src;
        for (int i=0; i < dst.size();i++) {
            if (dst[i].file.getBool("FTE"))
                dst[i].file.fte = true;
        }
        writer->writeBlock(dst);
    }

    void train(const TemplateList& data)
    {
        (void) data;
    }
    ;
    void init()
    {
        writer = QSharedPointer<Gallery>(Gallery::make(outputString));
    }

    QSharedPointer<Gallery> writer;
public:
    GalleryOutputTransform() : TimeVaryingTransform(false,false) {}
};

BR_REGISTER(Transform, GalleryOutputTransform)

} // namespace br

#include "io/galleryoutput.moc"
