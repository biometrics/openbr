#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/common.h>
#include <openbr/core/opencvutils.h>
#include <openbr/core/eval.h>

namespace br
{

typedef QPair<float,int> Pair;

/*!
 * \ingroup outputs
 * \brief Outputs the k-Nearest Neighbors from the gallery for each probe.
 * \author Ben Klein \cite bhklein
 */
class knnOutput : public Output
{
    Q_OBJECT

    int rowBlock, columnBlock;
    size_t headerSize, k;
    cv::Mat blockScores;

    ~knnOutput()
    {
        writeBlock();
    }

    void setBlock(int rowBlock, int columnBlock)
    {
        if ((rowBlock == 0) && (columnBlock == 0)) {
            k = file.get<size_t>("k", 20);
            QFile f(file);
            if (!f.open(QFile::WriteOnly))
                qFatal("Unable to open %s for writing.", qPrintable(file));
            size_t querySize = (size_t)queryFiles.size();
            f.write((const char*) &querySize, sizeof(size_t));
            f.write((const char*) &k, sizeof(size_t));
            headerSize = 2 * sizeof(size_t);
        } else {
            writeBlock();
        }

        this->rowBlock = rowBlock;
        this->columnBlock = columnBlock;

        int matrixRows  = std::min(static_cast<int>(queryFiles.size())-rowBlock*this->blockRows, blockRows);
        int matrixCols  = std::min(static_cast<int>(targetFiles.size())-columnBlock*this->blockCols, blockCols);

        blockScores = cv::Mat(matrixRows, matrixCols, CV_32FC1);
    }

    void setRelative(float value, int i, int j)
    {
        blockScores.at<float>(i,j) = value;
    }

    void set(float value, int i, int j)
    {
        (void) value; (void) i; (void) j;
        qFatal("Logic error.");
    }

    void writeBlock()
    {
        QFile f(file);
        if (!f.open(QFile::ReadWrite))
            qFatal("Unable to open %s for modifying.", qPrintable(file));
        QVector<Candidate> neighbors; neighbors.reserve(k * blockScores.rows);

        for (int i=0; i<blockScores.rows; i++) {
            size_t rank = 0;
            foreach (const Pair &pair, Common::Sort(OpenCVUtils::matrixToVector<float>(blockScores.row(i)), true)) {
                if (QString(targetFiles[pair.second]) != QString(queryFiles[rowBlock*this->blockRows+i])) {
                    Candidate candidate((size_t)pair.second, pair.first);
                    neighbors.push_back(candidate);
                    if (++rank >= k) break;
                }
            }
        }
        f.seek(headerSize + sizeof(Candidate)*quint64(rowBlock*this->blockRows)*k);
        f.write((const char*) neighbors.data(), blockScores.rows * k * sizeof(Candidate));
        f.close();
    }
};

BR_REGISTER(Output, knnOutput)

} // namespace br

#include "output/knn.moc"
