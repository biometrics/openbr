#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Write all mats to disk as images.
 * \author Brendan Klare \cite bklare
 */
class WriteTransform : public TimeVaryingTransform
{
    Q_OBJECT
    Q_PROPERTY(QString outputDirectory READ get_outputDirectory WRITE set_outputDirectory RESET reset_outputDirectory STORED false)
    Q_PROPERTY(QString underscore READ get_underscore WRITE set_underscore RESET reset_underscore STORED false)
    Q_PROPERTY(QString imgExtension READ get_imgExtension WRITE set_imgExtension RESET reset_imgExtension STORED false)
    Q_PROPERTY(int padding READ get_padding WRITE set_padding RESET reset_padding STORED false)
    BR_PROPERTY(QString, outputDirectory, "Temp")
    BR_PROPERTY(QString, underscore, "")
    BR_PROPERTY(QString, imgExtension, "jpg")
    BR_PROPERTY(int, padding, 5)

    int cnt;

    void init() {
        cnt = 0;
        if (! QDir(outputDirectory).exists())
            QDir().mkdir(outputDirectory);
    }

    void projectUpdate(const Template &src, Template &dst)
    {
        dst = src;
        QString path = QString("%1/image%2%3.%4").arg(outputDirectory).arg(cnt++, padding, 10, QChar('0')).arg(underscore.isEmpty() ? "" : "_" + underscore).arg(imgExtension);
        OpenCVUtils::saveImage(dst.m(), path);
    }

};

BR_REGISTER(Transform, WriteTransform)

} // namespace br

#include "io/write.moc"
