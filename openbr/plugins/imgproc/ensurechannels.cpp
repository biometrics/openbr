#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Enforce the matrix has a certain number of channels by adding or removing channels.
 * \author Josh Klontz \cite jklontz
 */
class EnsureChannelsTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int n READ get_n WRITE set_n RESET reset_n STORED false)
    BR_PROPERTY(int, n, 1)

    void project(const Template &src, Template &dst) const
    {
        if (src.m().channels() == n) {
            dst = src;
        } else {
            std::vector<Mat> mv;
            split(src, mv);

            // Add extra channels
            while ((int)mv.size() < n) {
                for (int i=0; i<src.m().channels(); i++) {
                    mv.push_back(mv[i]);
                    if ((int)mv.size() == n)
                        break;
                }
            }

            // Remove extra channels
            while ((int)mv.size() > n)
                mv.pop_back();

            merge(mv, dst);
        }
    }
};

BR_REGISTER(Transform, EnsureChannelsTransform)

} // namespace br

#include "ensurechannels.moc"
