#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Collect nearest neighbors and append them to metadata.
 * \author Charles Otto \cite caotto
 */
class CollectNNTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    Q_PROPERTY(int keep READ get_keep WRITE set_keep RESET reset_keep STORED false)
    BR_PROPERTY(int, keep, 20)

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;
        dst.clear();
        dst.m() = cv::Mat();
        Neighbors neighbors;
        for (int i=0; i < src.m().cols;i++) {
            // skip self compares
            if (i == src.file.get<int>("FrameNumber"))
                continue;
            neighbors.append(Neighbor(i, src.m().at<float>(0,i)));
        }
        int actuallyKeep = std::min(keep, neighbors.size());
        std::partial_sort(neighbors.begin(), neighbors.begin()+actuallyKeep, neighbors.end(), compareNeighbors);

        Neighbors selected = neighbors.mid(0, actuallyKeep);
        dst.file.set("neighbors", QVariant::fromValue(selected));
    }
};

BR_REGISTER(Transform, CollectNNTransform)

} // namespace br

#include "cluster/collectnn.moc"
