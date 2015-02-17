#include <QCoreApplication>
#include <QProcess>

#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Implements the YouTubesFaceDB \cite wolf11 experimental protocol.
 * \author Josh Klontz \cite jklontz
 */
class YouTubeFacesDBTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString algorithm READ get_algorithm WRITE set_algorithm RESET reset_algorithm STORED false)
    BR_PROPERTY(QString, algorithm, "")

    void project(const Template &src, Template &dst) const
    {
        static QMutex mutex;

        // First input is the header in 'splits.txt'
        if (src.file.get<int>("Index") == 0) return;

        const QStringList words = src.file.name.split(", ");
        const QString matrix = "YTF-"+algorithm+"/"+words[0] + "_" + words[1] + "_" + words[4] + ".mtx";
        const QStringList arguments = QStringList() << "-algorithm" << algorithm
                                                    << "-parallelism" << QString::number(Globals->parallelism)
                                                    << "-path" << Globals->path
                                                    << "-compare" << File(words[2]).resolved() << File(words[3]).resolved() << matrix;
        mutex.lock();
        int result = 0;
        if (!QFileInfo(matrix).exists())
            result = QProcess::execute(QCoreApplication::applicationFilePath(), arguments);
        mutex.unlock();

        if (result != 0)
            qWarning("Process for computing %s returned %d.", qPrintable(matrix), result);
        dst = Template();
    }
};

BR_REGISTER(Transform, YouTubeFacesDBTransform)

} // namespace br

#include "io/youtubefacesdb.moc"
