#include <QCoreApplication>
#include <QProcess>
#include <openbr_plugin.h>

#include "core/common.h"

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

    static void sort(TemplateList &templates)
    {
        // The file names in the YouTube Faces Database make it very difficult
        // for them to be ordered by frame number automatically,
        // hence we do it manually here.
        QList<int> frames;
        foreach (const Template &t, templates) {
            QStringList words = t.file.name.split('.');
            frames.append(words[words.size()-2].toInt());
        }

        typedef QPair<int,int> SortedFrame; // <frame number, original index>
        QList<SortedFrame> sortedFrames = Common::Sort(frames);
        TemplateList sortedTemplates; sortedTemplates.reserve(templates.size());
        foreach (const SortedFrame &sortedFrame, sortedFrames)
            sortedTemplates.append(templates[sortedFrame.second]);

        templates = sortedTemplates;
    }
};

BR_REGISTER(Transform, YouTubeFacesDBTransform)

} // namespace br

#include "youtube.moc"
