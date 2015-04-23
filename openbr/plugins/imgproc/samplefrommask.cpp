#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Samples pixels from a mask.
 * \author Scott Klum \cite sklum
 */
class SampleFromMaskTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(int minIndices READ get_minIndices WRITE set_minIndices RESET reset_minIndices STORED false)
    BR_PROPERTY(int, minIndices, 0)

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        Mat mask = src.file.get<Mat>("Mask");
        Mat indices;
        findNonZero(mask,indices);

        if (indices.total() > (size_t)minIndices) {
            Mat masked;
            src.m().copyTo(masked, mask);
            if (src.m().channels() > 1) {
                dst.m() = Mat(3,indices.total(),CV_32FC1);
                for (size_t j=0; j<indices.total(); j++) {
                    Vec3b v =  masked.at<Vec3b>(indices.at<Point>(j).y,indices.at<Point>(j).x);
                    dst.m().at<float>(0,j) = v[0];
                    dst.m().at<float>(1,j) = v[1];
                    dst.m().at<float>(2,j) = v[2];
                }
            } else {
                dst.m() = Mat(1,indices.total(),src.m().type());

                for (size_t j=0; j<indices.total(); j++)
                    dst.m().at<uchar>(0,j) = masked.at<uchar>(indices.at<Point>(j).y,indices.at<Point>(j).x);
            }
        } else {
            dst.file.fte = true;
            dst.file.remove("Mask");
            qWarning("No mask content for %s.",qPrintable(src.file.baseName()));
        }
    }
};

BR_REGISTER(Transform, SampleFromMaskTransform)

} // namespace br

#include "imgproc/samplefrommask.moc"
