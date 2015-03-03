#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV's adaptive thresholding.
 * \author Scott Klum \cite sklum
 */
class ThresholdTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_ENUMS(Type)
    Q_PROPERTY(Type type READ get_type WRITE set_type RESET reset_type STORED false)
    Q_PROPERTY(bool otsu READ get_otsu WRITE set_otsu RESET reset_otsu STORED false)
    Q_PROPERTY(int thresh READ get_thresh WRITE set_thresh RESET reset_thresh STORED false)
    Q_PROPERTY(int maxValue READ get_maxValue WRITE set_maxValue RESET reset_maxValue STORED false)

public:
    enum Type { Binary = THRESH_BINARY,
                BinaryInv = THRESH_BINARY_INV,
                Trunc = THRESH_TRUNC,
                ToZero = THRESH_TOZERO,
                ToZeroInv = THRESH_TOZERO_INV};

private:
    BR_PROPERTY(Type, type, Binary)
    BR_PROPERTY(bool, otsu, false)
    BR_PROPERTY(int, thresh, 0)
    BR_PROPERTY(int, maxValue, 255)

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        Mat mask;
        threshold(src, mask, thresh, maxValue, otsu ? type+THRESH_OTSU : type);

        dst.file.set("Mask",QVariant::fromValue(mask));
    }
};

BR_REGISTER(Transform, ThresholdTransform)

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
