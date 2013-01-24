#include <openbr_plugin.h>

#include "core/common.h"

using namespace br;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Impostor Uniqueness Measure \cite klare12
 * \author Josh Klontz \cite jklontz
 */
class IUMTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(br::Distance* distance READ get_distance WRITE set_distance RESET reset_distance)
    BR_PROPERTY(br::Distance*, distance, Factory<Distance>::make(".Dist(L2)"))
    br::TemplateList impostors;

    void train(const TemplateList &data)
    {
        distance->train(data);
        impostors = data;
    }

    void project(const Template &src, Template &dst) const
    {
        QList<float> scores = distance->compare(impostors, src);
        float min, max;
        Common::MinMax(scores, &min, &max);
        double mean;
        Common::Mean(scores, &mean);
        dst = src;
        dst.file.insert("IUM", float((max-mean)/(max-min)));
    }

    void store(QDataStream &stream) const
    {
        distance->store(stream);
        stream << impostors;
    }

    void load(QDataStream &stream)
    {
        distance->load(stream);
        stream >> impostors;
    }
};

BR_REGISTER(Transform, IUMTransform)

} // namespace br

#include "quality.moc"
