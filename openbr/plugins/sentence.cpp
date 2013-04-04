#include <stdint.h>
#include "openbr_internal.h"

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Ordered words
 * \author Josh Klontz \cite jklontz
 */
class SentenceTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        QByteArray sentence;
        QDataStream stream(&sentence, QIODevice::WriteOnly);
        for (int i=0; i<src.size(); i++) {
            const Mat &m = src[i];
            if (!m.data) continue;
            stream.writeRawData((const char*)&i, 4);
            stream.writeRawData((const char*)&m.rows, 4);
            stream.writeRawData((const char*)&m.cols, 4);
            stream.writeRawData((const char*)m.data, 4*m.rows*m.cols);
        }
        dst.file = src.file;
        dst.m() = Mat(1, sentence.size(), CV_8UC1, sentence.data()).clone();
    }
};

BR_REGISTER(Transform, SentenceTransform)

/*!
 * \ingroup distances
 * \brief Distance between sentences
 * \author Josh Klontz \cite jklontz
 */
class SentenceSimilarityDistance : public Distance
{
    Q_OBJECT

    float compare(const Template &a, const Template &b) const
    {
        uchar *aBuffer = a.m().data;
        uchar *bBuffer = b.m().data;
        const uchar *aEnd = aBuffer + a.m().cols;
        const uchar *bEnd = bBuffer + b.m().cols;

        int32_t aWord, bWord, aRows, bRows, aColumns, bColumns;
        float *aData, *bData;
        aWord = aRows = aColumns = -2;
        bWord = bRows = bColumns = -1;
        aData = bData = NULL;

        float distance = 0;
        int comparisons = 0;
        while (true) {
            if (aWord < bWord) {
                if (aBuffer == aEnd) return distance == 0 ? -std::numeric_limits<float>::max() : comparisons / distance;
                aWord = *reinterpret_cast<int32_t*>(aBuffer);
                aRows = *reinterpret_cast<int32_t*>(aBuffer+4);
                aColumns = *reinterpret_cast<int32_t*>(aBuffer+8);
                aData = reinterpret_cast<float*>(aBuffer+12);
                aBuffer += 12 + 4*aRows*aColumns;
            } else if (bWord < aWord) {
                if (bBuffer == bEnd) return comparisons == 0 ? -std::numeric_limits<float>::max() : comparisons / distance;
                bWord = *reinterpret_cast<int32_t*>(bBuffer);
                bRows = *reinterpret_cast<int32_t*>(bBuffer+4);
                bColumns = *reinterpret_cast<int32_t*>(bBuffer+8);
                bData = reinterpret_cast<float*>(bBuffer+12);
                bBuffer += 12 + 4*bRows*bColumns;
            } else {
                for (int i=0; i<aRows; i++)
                    for (int j=0; j<bRows; j++)
                        for (int k=0; k<aColumns; k++)
                            distance += pow(aData[i*aColumns+k] - bData[j*bColumns+k], 2.f);
                comparisons += aRows * bRows * aColumns;
                aWord = -2;
                bWord = -1;
            }
        }
    }
};

BR_REGISTER(Distance, SentenceSimilarityDistance)

} // namespace br

#include "sentence.moc"
