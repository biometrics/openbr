#include <openbr/plugins/openbr_internal.h>

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

        int w = mask.rows;
        int h = mask.cols;
        int left = w;
        int right = 0;
        int top = h;
        int bottom = 0;
        for (int i = 0 ; i < w; i++) {
            for (int j = 0 ; j < h; j++) {
                if (mask.at<unsigned char>(i,j)) {
                    if (i < left)
                        left = i;
                    if (i > right)
                        right = i;
                    if (j < top)
                        top = j;
                    if (j > bottom)
                        bottom = j;
                }
            }
        }

        if (fixedAspectRatio) {
            h = bottom - top + 1;
            w = right - left + 1;
            if (h > w) {
                int h2 = (h - w) / 2;
                right += h2;
                left -= h2;
            } else {
                int w2 = (w - h) / 2;
                bottom += w2;
                top -= w2;
            }
        }

        dst.m() = Mat(src.m(), Rect(top, left, bottom - top + 1, right - left + 1));

    }
};

BR_REGISTER(Transform, CropFromMaskTransform)

} // namespace br

#include "imgproc/threshold.moc"
