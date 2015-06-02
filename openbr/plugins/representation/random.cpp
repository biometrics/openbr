#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>
#include <openbr/core/common.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup galleries
 * \brief Computes first order gradient histogram features using an integral image
 * \author Scott Klum \cite sklum
 */
class RandomRepresentation : public Representation
{
    Q_OBJECT
    Q_PROPERTY(br::Representation *representation READ get_representation WRITE set_representation RESET reset_representation STORED false)
    Q_PROPERTY(int count READ get_count WRITE set_count RESET reset_count STORED false)
    BR_PROPERTY(br::Representation *, representation, NULL)
    BR_PROPERTY(int, count, 20000)

    void init()
    {
        representation->init();

        const int nFeatures = representation->numFeatures();

        features = Common::RandSample(count,nFeatures);
    }

    void preprocess(const Mat &src, Mat &dst) const
    {
        representation->preprocess(src,dst);
    }

    float evaluate(const Mat &image, int idx) const
    {
        return representation->evaluate(image,features[idx]);
    }

    Mat evaluate(const Mat &image, const QList<int> &indices) const
    {
        QList<int> newIndices;
        if (indices.empty())
            newIndices = features;
        else
            for (int i = 0; i < indices.size(); i++)
                newIndices.append(features[indices[i]]);

        return representation->evaluate(image,newIndices);
    }

    int numFeatures() const
    {
        return features.size();
    }

    int numChannels() const
    {
        return representation->numChannels();
    }

    Size windowSize(int *dx, int *dy) const
    {
        return representation->windowSize(dx,dy);
    }

    int maxCatCount() const
    {
        return representation->maxCatCount();
    }

    void load(QDataStream &stream)
    {
        stream >> features;

        representation->load(stream);
    }

    void store(QDataStream &stream) const
    {
        stream << features;

        representation->load(stream);
    }

    QList<int> features;
};

BR_REGISTER(Representation, RandomRepresentation)

} // namespace br

#include "representation/random.moc"



