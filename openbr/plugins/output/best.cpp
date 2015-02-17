#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/qtutils.h>

namespace br
{

/*!
 * \ingroup outputs
 * \brief The highest scoring matches.
 * \author Josh Klontz \cite jklontz
 */
class bestOutput : public Output
{
    Q_OBJECT

    typedef QPair< float, QPair<int, int> > BestMatch;
    QList<BestMatch> bestMatches;

    ~bestOutput()
    {
        if (file.isNull() || bestMatches.isEmpty()) return;
        qSort(bestMatches);
        QStringList lines; lines.reserve(bestMatches.size()+1);
        lines.append("Value,Target,Query");
        for (int i=bestMatches.size()-1; i>=0; i--)
            lines.append(QString::number(bestMatches[i].first) + "," + targetFiles[bestMatches[i].second.second] + "," + queryFiles[bestMatches[i].second.first]);
        QtUtils::writeFile(file, lines);
    }

    void initialize(const FileList &targetFiles, const FileList &queryFiles)
    {
        Output::initialize(targetFiles, queryFiles);
        bestMatches.reserve(queryFiles.size());
        for (int i=0; i<queryFiles.size(); i++)
            bestMatches.append(BestMatch(-std::numeric_limits<float>::max(), QPair<int,int>(-1, -1)));
    }

    void set(float value, int i, int j)
    {
        static QMutex lock;

        // Return early for self similar matrices
        if (selfSimilar && (i == j)) return;

        if (value > bestMatches[i].first) {
            lock.lock();
            if (value > bestMatches[i].first)
                bestMatches[i] = BestMatch(value, QPair<int,int>(i,j));
            lock.unlock();
        }
    }
};

BR_REGISTER(Output, bestOutput)

} // namespace br

#include "output/best.moc"
