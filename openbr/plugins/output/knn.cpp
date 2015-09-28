#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/common.h>
#include <openbr/core/opencvutils.h>
#include <openbr/core/eval.h>

namespace br
{

/*!
 * \ingroup outputs
 * \brief Outputs the k-Nearest Neighbors from the gallery for each probe.
 * \author Ben Klein \cite bhklein
 */
class knnOutput : public MatrixOutput
{
    Q_OBJECT

    ~knnOutput()
    {
        size_t num_probes = (size_t)queryFiles.size();
        if (targetFiles.isEmpty() || queryFiles.isEmpty()) return;
        size_t k = file.get<size_t>("k", 20);

        if ((size_t)targetFiles.size() < k)
            qFatal("Gallery size %s is smaller than k = %s.", qPrintable(QString::number(targetFiles.size())), qPrintable(QString::number(k)));

        QFile f(file);
        if (!f.open(QFile::WriteOnly))
            qFatal("Unable to open %s for writing.", qPrintable(file));
        f.write((const char*) &num_probes, sizeof(size_t));
        f.write((const char*) &k, sizeof(size_t));

        QVector<Candidate> neighbors; neighbors.reserve(num_probes*k);

        for (size_t i=0; i<num_probes; i++) {
            typedef QPair<float,int> Pair;
            size_t rank = 0;
            foreach (const Pair &pair, Common::Sort(OpenCVUtils::matrixToVector<float>(data.row(i)), true)) {
                if (QString(targetFiles[pair.second]) != QString(queryFiles[i])) {
                    Candidate candidate((size_t)pair.second, pair.first);
                    neighbors.push_back(candidate);
                    if (++rank >= k) break;
                }
            }
        }
        f.write((const char*) neighbors.data(), num_probes * k * sizeof(Candidate));
        f.close();
    }
};

BR_REGISTER(Output, knnOutput)

} // namespace br

#include "output/knn.moc"
