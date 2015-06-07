#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>
#include <openbr/core/common.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup representations
 * \brief A meta Representation that creates a smaller, random feature space from the full feature space of a given representation.
 * \author Scott Klum \cite sklum
 * \br_property Representation* representation The representation from which to create the random feature space
 * \br_property int count The size of the random feature space
 */
class RandomRepresentation : public Representation
{
    Q_OBJECT

    Q_PROPERTY(br::Representation* representation READ get_representation WRITE set_representation RESET reset_representation STORED false)
    Q_PROPERTY(int count READ get_count WRITE set_count RESET reset_count STORED false)
    BR_PROPERTY(br::Representation*, representation, NULL)
    BR_PROPERTY(int, count, 20000)

    QList<int> features;

    void train(const QList<Mat> &images, const QList<float> &labels)
    {
        representation->train(images, labels);

        const int nFeatures = representation->numFeatures();

        if (Globals->verbose)
            qDebug() << "Randomly sampling from" << nFeatures << "features.";

        features = Common::RandSample(count,nFeatures,0,true);
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
        representation->load(stream);

        int numFeatures; stream >> numFeatures;
        for (int i=0; i<numFeatures; i++) {
            int feature; stream >> feature;
            features.append(feature);
        }
    }

    void store(QDataStream &stream) const
    {
        representation->store(stream);

        stream << features.size();
        for (int i=0; i<features.size(); i++)
            stream << features[i];
    }
};

BR_REGISTER(Representation, RandomRepresentation)

} // namespace br

#include "representation/random.moc"



