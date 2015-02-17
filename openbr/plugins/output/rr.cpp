#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/common.h>
#include <openbr/core/qtutils.h>
#include <openbr/core/opencvutils.h>

namespace br
{

/*!
 * \ingroup outputs
 * \brief Rank retrieval output.
 * \author Josh Klontz \cite jklontz
 * \author Scott Klum \cite sklum
 */
class rrOutput : public MatrixOutput
{
    Q_OBJECT

    ~rrOutput()
    {
        if (file.isNull() || targetFiles.isEmpty() || queryFiles.isEmpty()) return;
        const int limit = file.get<int>("limit", 20);
        const bool byLine = file.getBool("byLine");
        const bool simple = file.getBool("simple");
        const float threshold = file.get<float>("threshold", -std::numeric_limits<float>::max());

        QStringList lines;

        for (int i=0; i<queryFiles.size(); i++) {
            QStringList files;
            if (simple) files.append(queryFiles[i].fileName());

            typedef QPair<float,int> Pair;
            foreach (const Pair &pair, Common::Sort(OpenCVUtils::matrixToVector<float>(data.row(i)), true, limit)) {
                if (Globals->crossValidate > 0 ? (targetFiles[pair.second].get<int>("Partition",-1) == -1 || targetFiles[pair.second].get<int>("Partition",-1) == queryFiles[i].get<int>("Partition",-1)) : true) {
                    if (pair.first < threshold) break;
                    File target = targetFiles[pair.second];
                    target.set("Score", QString::number(pair.first));
                    if (simple) files.append(target.fileName() + " " + QString::number(pair.first));
                    else files.append(target.flat());
                }
            }
            lines.append(files.join(byLine ? "\n" : ","));
        }

        QtUtils::writeFile(file, lines);
    }
};

BR_REGISTER(Output, rrOutput)

} // namespace br

#include "output/rr.moc"
