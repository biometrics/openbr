#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Normalize matrix to unit length
 * \author Josh Klontz \cite jklontz
 */
class NormalizeTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_ENUMS(NormType)
    Q_PROPERTY(NormType normType READ get_normType WRITE set_normType RESET reset_normType STORED false)

    Q_PROPERTY(bool ByRow READ get_ByRow WRITE set_ByRow RESET reset_ByRow STORED false)
    BR_PROPERTY(bool, ByRow, false)
    Q_PROPERTY(int alpha READ get_alpha WRITE set_alpha RESET reset_alpha STORED false)
    BR_PROPERTY(int, alpha, 1)
    Q_PROPERTY(int beta READ get_beta WRITE set_beta RESET reset_beta STORED false)
    BR_PROPERTY(int, beta, 0)
    Q_PROPERTY(bool squareRoot READ get_squareRoot WRITE set_squareRoot RESET reset_squareRoot STORED false)
    BR_PROPERTY(bool, squareRoot, false)

public:
    /*!< */
    enum NormType { Inf = NORM_INF,
                    L1 = NORM_L1,
                    L2 = NORM_L2,
                    Range = NORM_MINMAX };

private:
    BR_PROPERTY(NormType, normType, L2)

    static void signedSquareRoot(Mat &m)
    {
        for (int i=0; i<m.rows; i++)
            for (int j=0; j<m.cols; j++) {
                float &val = m.at<float>(i, j);
                val = sqrtf(fabsf(val)) * (val >= 0 ? 1 : -1);
            }
    }

    void project(const Template &src, Template &dst) const
    {
        if (!ByRow) {
            normalize(src, dst, alpha, beta, normType, CV_32F);
            if (squareRoot)
                signedSquareRoot(dst);
        }

        else {
            dst = src;
            for (int i=0; i<dst.m().rows; i++) {
                Mat temp;
                normalize(dst.m().row(i), temp, alpha, beta, normType);
                if (squareRoot)
                    signedSquareRoot(temp);
                temp.copyTo(dst.m().row(i));
            }
        }

    }
};

BR_REGISTER(Transform, NormalizeTransform)

} // namespace br

#include "imgproc/normalize.moc"
