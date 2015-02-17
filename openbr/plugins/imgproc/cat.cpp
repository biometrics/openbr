#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Concatenates all input matrices into a single matrix.
 * \author Josh Klontz \cite jklontz
 */
class CatTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(int partitions READ get_partitions WRITE set_partitions RESET reset_partitions)
    BR_PROPERTY(int, partitions, 1)

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;

        if (src.size() % partitions != 0)
            qFatal("%d partitions does not evenly divide %d matrices.", partitions, src.size());
        QVector<int> sizes(partitions, 0);
        for (int i=0; i<src.size(); i++)
            sizes[i%partitions] += src[i].total();

        if (!src.empty())
            foreach (int size, sizes)
                dst.append(Mat(1, size, src.m().type()));

        QVector<int> offsets(partitions, 0);
        for (int i=0; i<src.size(); i++) {
            size_t size = src[i].total() * src[i].elemSize();
            int j = i % partitions;
            memcpy(&dst[j].data[offsets[j]], src[i].ptr(), size);
            offsets[j] += size;
        }
    }
};

BR_REGISTER(Transform, CatTransform)

} // namespace br

#include "imgproc/cat.moc"
