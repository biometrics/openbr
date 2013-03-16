#include <openbr_plugin.h>

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
            stream << i << m.rows << m.cols;
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
        const int aSize = a.m().cols;
        const int bSize = b.m().cols;
        uchar *aData = a.m().data;
        uchar *bData = b.m().data;
        const uchar *aEnd = aData + aSize;
        const uchar *bEnd = bData + bSize;

        int32_t aWord, bWord, aRows, bRows, aColumns, bColumns;
        aWord = aRows = aColumns = -2;
        bWord = bRows = bColumns = -1;
        float distance = 0;
        int comparisons = 0;
        while ((aData != aEnd) && (bData != bEnd)) {
            if (aWord < bWord) {
                aWord = *reinterpret_cast<int32_t*>(aData);
                aRows = *reinterpret_cast<int32_t*>(aData+4);
                aColumns = *reinterpret_cast<int32_t*>(aData+8);
                aData += 12;
            } else if (bWord < aWord) {
                bWord = *reinterpret_cast<int32_t*>(bData);
                bRows = *reinterpret_cast<int32_t*>(bData+4);
                bColumns = *reinterpret_cast<int32_t*>(bData+8);
                bData += 12;
            } else {
                for (int i=0; i<aRows; i++)
                    for (int j=0; j<bRows; j++)
                        for (int k=0; k<aColumns; k++)
                            distance += *reinterpret_cast<float*>(aData+4*k) * *reinterpret_cast<float*>(bData+4*k);
                comparisons += aRows * bRows;
            }
        }

        return comparisons / distance;
    }
};

BR_REGISTER(Distance, SentenceSimilarityDistance)

} // namespace br

#include "sentence.moc"
