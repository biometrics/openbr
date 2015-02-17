#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Prediction using only the red wavelength; magic numbers from jmp
 * \author E. Taborsky \cite mmtaborsky
 */
class RedLinearRegressionTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        Mat m; src[0].convertTo(m, CV_32F); assert(m.isContinuous() && (m.channels() == 1));

        const float rmult = .6533673;
        const float add = 41.268;

        Mat dst1(m.size(), CV_32F);
        int rows = m.rows;
        int cols = m.cols;

        const float *rsrc = (const float*) m.ptr();
        float *p = (float*)dst1.ptr();

        for (int r = 0; r < rows; r++){
            for (int c = 0; c < cols; c++){
                int index = r*cols+c;
                const float rval = rsrc[index];
                p[index] = rval*rmult+add;
            }
        }
        dst = dst1;
    }
};

BR_REGISTER(Transform, RedLinearRegressionTransform)

} // namespace br

#include "imgproc/redlinearregression.moc"
