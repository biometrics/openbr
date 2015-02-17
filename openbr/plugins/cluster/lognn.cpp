#include <fstream>

#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Log nearest neighbors to specified file.
 * \author Charles Otto \cite caotto
 */
class LogNNTransform : public TimeVaryingTransform
{
    Q_OBJECT

    Q_PROPERTY(QString fileName READ get_fileName WRITE set_fileName RESET reset_fileName STORED false)
    BR_PROPERTY(QString, fileName, "")

    std::fstream fout;

    void projectUpdate(const Template &src, Template &dst)
    {
        dst = src;

        if (!dst.file.contains("neighbors")) {
            fout << std::endl;
            return;
        }

        Neighbors neighbors = dst.file.get<Neighbors>("neighbors");
        if (neighbors.isEmpty() ) {
            fout << std::endl;
            return;
        }

        QString aLine;
        aLine.append(QString::number(neighbors[0].first)+":"+QString::number(neighbors[0].second));
        for (int i=1; i < neighbors.size();i++)
            aLine.append(","+QString::number(neighbors[i].first)+":"+QString::number(neighbors[i].second));

        fout << qPrintable(aLine) << std::endl;
    }

    void init()
    {
        if (!fileName.isEmpty())
            fout.open(qPrintable(fileName), std::ios_base::out);
    }

    void finalize(TemplateList &output)
    {
        (void) output;
        fout.close();
    }

public:
    LogNNTransform() : TimeVaryingTransform(false, false) {}
};

BR_REGISTER(Transform, LogNNTransform)

} // namespace br

#include "cluster/lognn.moc"
