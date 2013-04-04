#include "openbr_internal.h"
#include <LatentEFS.h>

// Necessary to allocate a large memory though the actual template size may be much smaller
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
        Globals->abbreviations.insert("NECLatentLFFS1", "Open+NECLatent1Enroll(true,ELFT_M):NECLatent1Compare(ELFT_M)");
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
                     ELFT,
                     ELFT_M };

private:
    BR_PROPERTY(bool, latent, false)
    BR_PROPERTY(Algorithm, algorithm, LFML)

    void project(const Template &src, Template &dst) const
    {
        if (src.m().type() != CV_8UC1) qFatal("Requires 8UC1 data!");
        uchar *data = src.m().data;
        const int rows = src.m().rows;
        const int columns = src.m().cols;
        uchar buff[MAX_TEMPLATE_SIZE];
        uchar* pBuff = NULL;
        int size = 0;
        int error;

        if (latent) {
            if      (algorithm == LFML) error = NEC_LFML_ExtractLatent(data, rows, columns, 500, buff, &size);
            else if (algorithm == ELFT) error = NEC_ELFT_ExtractLatent(data, rows, columns, 500, 4, buff, &size);
            else                        error = NEC_ELFT_M_ExtractLatent(data, columns, 1, &pBuff, &size);
        } else {
            if      (algorithm == LFML) error = NEC_LFML_ExtractTenprint(data, rows, columns, 500, buff, &size);
            else if (algorithm == ELFT) error = NEC_ELFT_ExtractTenprint(data, rows, columns, 500, 2, buff, &size);
            else                        qFatal("ELFT_M Tenprint not implemented.");
        }

        if (!error) {
            cv::Mat n(1, size, CV_8UC1);
            memcpy(n.data, buff, size);
            dst.m() = n;
        } else {
            qWarning("NECLatent1EnrollTransform error %d for file %s.", error, qPrintable(src.file.flat()));
            dst.m() = cv::Mat();
            dst.file.set("FTE", true);
        }

        if (pBuff != NULL) NEC_ELFT_M_FreeTemplate(&pBuff);
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
                     ELFT,
                     ELFT_M };

private:
    BR_PROPERTY(Algorithm, algorithm, LFML)

    float compare(const Template &a, const Template &b) const
    {
        uchar *aData = a.m().data;
        uchar *bData = b.m().data;
        if (!a.m().data || !b.m().data) return -std::numeric_limits<float>::max();
        int score;
        if      (algorithm == LFML) NEC_LFML_Verify(bData, b.m().total(), aData, a.m().total(), &score);
        else if (algorithm == ELFT) NEC_ELFT_Verify(bData, aData, &score, 1);
        else                        NEC_ELFT_M_Verify(bData, aData, &score, 1);
        return score;
    }
};

BR_REGISTER(Distance, NECLatent1CompareDistance)

} // namespace br

#include "neclatent1.moc"
