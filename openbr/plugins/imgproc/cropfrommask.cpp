#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/common.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Crops image based on mask metadata
 * \author Brendan Klare \cite bklare
 */
class CropFromMaskTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(bool fixedAspectRatio READ get_fixedAspectRatio WRITE set_fixedAspectRatio RESET reset_fixedAspectRatio STORED false)
    BR_PROPERTY(bool, fixedAspectRatio, true)

private:

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        Mat mask = dst.file.get<Mat>("Mask");

        Mat indices;
        findNonZero(mask,indices);

        if (indices.total() > 0) {
            QList<int> x, y;
            for (size_t i=0; i<indices.total(); i++) {
                x.append(indices.at<Point>(i).x);
                y.append(indices.at<Point>(i).y);
            }

            int t, b, l, r;
            Common::MinMax(x,&l,&r);
            Common::MinMax(y,&t,&b);

            if (fixedAspectRatio) {
                int h, w;

                h = b - t + 1;
                w = r - l + 1;

                if (h > w) {
                    int h2 = (h - w) / 2;
                    r += h2;
                    l -= h2;
                } else {
                    int w2 = (w - h) / 2;
                    b += w2;
                    t -= w2;
                }
            }

            t = max(t,0);
            b = min(b, src.m().rows-1);
            l = max(l,0);
            r = min(r, src.m().cols-1);

            dst.m() = Mat(src.m(), Rect(l, t, r - l + 1, b - t + 1));
        } else {
            // Avoid serializing mask
            dst.file.remove("Mask");
            dst.file.fte = true;
        }
    }
};

BR_REGISTER(Transform, CropFromMaskTransform)

} // namespace br

#include "imgproc/cropfrommask.moc"
