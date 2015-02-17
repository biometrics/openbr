#include <opencv2/highgui/highgui.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Read images
 * \author Josh Klontz \cite jklontz
 */
class ReadTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_ENUMS(Mode)
    Q_PROPERTY(Mode mode READ get_mode WRITE set_mode RESET reset_mode)

public:
    enum Mode
    {
        Unchanged = IMREAD_UNCHANGED,
        Grayscale = IMREAD_GRAYSCALE,
        Color     = IMREAD_COLOR,
        AnyDepth  = IMREAD_ANYDEPTH,
        AnyColor  = IMREAD_ANYCOLOR
    };

private:
    BR_PROPERTY(Mode, mode, Color)

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;
        if (Globals->verbose)
            qDebug("Opening %s", qPrintable(src.file.flat()));

        if (src.empty()) {
            const Mat img = imread(src.file.resolved().toStdString(), mode);
            if (img.data) dst.append(img);
            else          dst.file.fte = true;
        } else {
            foreach (const Mat &m, src) {
                const Mat img = imdecode(m, mode);
                if (img.data) dst.append(img);
                else          dst.file.fte = true;
            }
        }
        if (dst.file.fte)
            qWarning("Error opening %s", qPrintable(src.file.flat()));
    }
};

BR_REGISTER(Transform, ReadTransform)

} // namespace br

#include "io/read.moc"
