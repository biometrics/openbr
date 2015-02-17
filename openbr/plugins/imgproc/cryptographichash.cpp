#include <QCryptographicHash>

#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Wraps QCryptographicHash
 * \author Josh Klontz \cite jklontz
 */
class CryptographicHashTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_ENUMS(Algorithm)
    Q_PROPERTY(Algorithm algorithm READ get_algorithm WRITE set_algorithm RESET reset_algorithm STORED false)

public:
    /*!< */
    enum Algorithm { Md4 = QCryptographicHash::Md4,
                     Md5 = QCryptographicHash::Md5,
                     Sha1 = QCryptographicHash::Sha1 };

private:
    BR_PROPERTY(Algorithm, algorithm, Md5)

    void project(const Template &src, Template &dst) const
    {
        const cv::Mat &m = src;
        QByteArray data((const char *)m.data, int(m.total()*m.elemSize()));
        QByteArray hash = QCryptographicHash::hash(data, (QCryptographicHash::Algorithm)algorithm);
        cv::Mat n(1, hash.size(), CV_8UC1);
        memcpy(n.data, hash.data(), hash.size());
        dst = Template(src.file, n);
    }
};

BR_REGISTER(Transform, CryptographicHashTransform)

} // namespace br

#include "imgproc/cryptographichash.moc"
