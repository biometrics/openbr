#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Remove the row-wise training set average.
 * \author Josh Klontz \cite jklontz
 */
class RowWiseMeanCenterTransform : public Transform
{
    Q_OBJECT
    Mat mean;

    void train(const TemplateList &data)
    {
        Mat m = OpenCVUtils::toMatByRow(data.data());
        mean = Mat(1, m.cols, m.type());
        for (int i=0; i<m.cols; i++)
            mean.col(i) = cv::mean(m.col(i));
    }

    void project(const Template &src, Template &dst) const
    {
        Mat m = src.m().clone();
        for (int i=0; i<m.rows; i++)
            m.row(i) -= mean;
        dst = m;
    }

    void store(QDataStream &stream) const
    {
        stream << mean;
    }

    void load(QDataStream &stream)
    {
        stream >> mean;
    }
};

BR_REGISTER(Transform, RowWiseMeanCenterTransform)

} // namespace br

#include "imgproc/rowwisemeancenter.moc"
