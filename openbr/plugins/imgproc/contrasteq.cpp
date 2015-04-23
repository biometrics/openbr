#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/tanh_sse.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Perform contrast equalization
 * \br_paper Xiaoyang Tan; Triggs, B.;
 *           "Enhanced Local Texture Feature Sets for Face Recognition Under Difficult Lighting Conditions,"
 *           Image Processing, IEEE Transactions on , vol.19, no.6, pp.1635-1650, June 2010
 * \author Josh Klontz \cite jklontz
 */
class ContrastEqTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(float a READ get_a WRITE set_a RESET reset_a STORED false)
    Q_PROPERTY(float t READ get_t WRITE set_t RESET reset_t STORED false)
    BR_PROPERTY(float, a, 1)
    BR_PROPERTY(float, t, 0.1)

    void project(const Template &src, Template &dst) const
    {
        if (src.m().channels() != 1) qFatal("Expected single channel source matrix.");

        // Stage 1
        Mat stage1;
        {
            Mat abs_dst;
            absdiff(src, Scalar(0), abs_dst);
            Mat pow_dst;
            pow(abs_dst, a, pow_dst);
            float denominator = pow((float)mean(pow_dst)[0], 1.f/a);
            src.m().convertTo(stage1, CV_32F, 1/denominator);
        }

        // Stage 2
        Mat stage2;
        {
            Mat abs_dst;
            absdiff(stage1, Scalar(0), abs_dst);
            Mat min_dst;
            min(abs_dst, t, min_dst);
            Mat pow_dst;
            pow(min_dst, a, pow_dst);
            float denominator = pow((float)mean(pow_dst)[0], 1.f/a);
            stage1.convertTo(stage2, CV_32F, 1/denominator);
        }

        // Hyperbolic tangent
        const int nRows = src.m().rows;
        const int nCols = src.m().cols;
        const float* p = (const float*)stage2.ptr();
        Mat m(nRows, nCols, CV_32FC1);
        for (int i=0; i<nRows; i++)
            for (int j=0; j<nCols; j++)
                m.at<float>(i, j) = fast_tanh(p[i*nCols+j]);
                // TODO: m.at<float>(i, j) = t * fast_tanh(p[i*nCols+j] / t);

        dst = m;
    }
};

BR_REGISTER(Transform, ContrastEqTransform)

} // namespace br

#include "imgproc/contrasteq.moc"
