#include <openbr_plugin.h>
#include <LatentEFS.h>

// necessary to allocate a large memory though the actual template size
// may be much smaller
#define MAX_TEMPLATE_SIZE 400000

namespace br
{

/*!
 * \ingroup initializers
 * \brief Initialize the NEC Latent SDK wrapper.
 * \author Josh Klontz \cite jklontz
 */
class NECLatent1Initialier : public Initializer
{
    Q_OBJECT

    void initialize() const
    {
        Globals->abbreviations.insert("NECTenprint1", "Open+Cvt(Gray)+NECLatent1Enroll:NECLatent1Compare");
        Globals->abbreviations.insert("NECLatent1", "Open+Cvt(Gray)+NECLatent1Enroll(true):NECLatent1Compare");
    }
};

BR_REGISTER(Initializer, NECLatent1Initialier)

/*!
 * \ingroup transforms
 * \brief Enroll an NEC latent fingerprint.
 * \author Josh Klontz \cite jklontz
 * \warning Applications using this transform must have their working directory be the 'bin/win/32' folder of the NEC Latent SDK.
 */
class NECLatent1EnrollTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_ENUMS(Algorithm)
    Q_PROPERTY(bool latent READ get_latent WRITE set_latent RESET reset_latent STORED false)
    Q_PROPERTY(Algorithm algorithm READ get_algorithm WRITE set_algorithm RESET reset_algorithm STORED false)

public:
    enum Algorithm { LFML,
                     ELFT };

private:
    BR_PROPERTY(bool, latent, false)
    BR_PROPERTY(Algorithm, algorithm, LFML)

    void project(const Template &src, Template &dst) const
    {
        if (src.m().type() != CV_8UC1) qFatal("Requires 8UC1 data!");
        unsigned char data[MAX_TEMPLATE_SIZE];
        int size = 0;
        int error;

        if (latent) {
            if (algorithm == LFML) error = NEC_LFML_ExtractLatent(src.m().data, src.m().rows, src.m().cols, 500, data, &size);
            else                   error = NEC_ELFT_ExtractLatent(src.m().data, src.m().rows, src.m().cols, 500, 4, data, &size);
        } else {
            if (algorithm == LFML) error = NEC_LFML_ExtractTenprint(src.m().data, src.m().rows, src.m().cols, 500, data, &size);
            else                   error = NEC_ELFT_ExtractTenprint(src.m().data, src.m().rows, src.m().cols, 500, 2, data, &size);
        }

        if (!error) {
            cv::Mat n(1, size, CV_8UC1);
            memcpy(n.data, data, size);
            dst.m() = n;
        } else {
            qWarning("NECLatent1EnrollTransform error %d for file %s.", error, qPrintable(src.file.flat()));
            dst.m() = cv::Mat();
            dst.file.set("FTE", true);
        }
    }
};

BR_REGISTER(Transform, NECLatent1EnrollTransform)

/*!
 * \ingroup distances
 * \brief Compare two NEC latent fingerprints
 * \author Josh Klontz \cite jklontz
 */
class NECLatent1CompareDistance : public Distance
{
    Q_OBJECT
    Q_ENUMS(Algorithm)
    Q_PROPERTY(Algorithm algorithm READ get_algorithm WRITE set_algorithm RESET reset_algorithm STORED false)

public:
    enum Algorithm { LFML,
                     ELFT };

private:
    BR_PROPERTY(Algorithm, algorithm, LFML)

    float compare(const Template &a, const Template &b) const
    {
        if (!a.m().data || !b.m().data) return -std::numeric_limits<float>::max();
        int score;
        if (algorithm == LFML) NEC_LFML_Verify(b.m().data, b.m().total(), a.m().data, a.m().total(), &score);
        else                   NEC_ELFT_Verify(b.m().data, a.m().data, &score, 2);
        return score;
    }
};

BR_REGISTER(Distance, NECLatent1CompareDistance)

} // namespace br

#include "neclatent1.moc"
