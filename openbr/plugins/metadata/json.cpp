#include <QJsonObject>
#include <QJsonDocument>

#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Represent the metadata as JSON template data.
 * \author Josh Klontz \cite jklontz
 */
class JSONTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;
        dst.file.set("AlgorithmID", 2);
        const QByteArray json = QJsonDocument(QJsonObject::fromVariantMap(dst.file.localMetadata())).toJson().replace('\n', ' ');
        dst += cv::Mat(1, json.size()+1 /*include null terminator*/, CV_8UC1, (void*) json.data()).clone();
    }
};

BR_REGISTER(Transform, JSONTransform)

} // namespace br

#include "metadata/json.moc"
