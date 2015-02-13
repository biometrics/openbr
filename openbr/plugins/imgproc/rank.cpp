#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/common.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Converts each element to its rank-ordered value.
 * \author Josh Klontz \cite jklontz
 */
class RankTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        const Mat &m = src;
        assert(m.channels() == 1);
        dst = Mat(m.rows, m.cols, CV_32FC1);
        typedef QPair<float,int> Tuple;
        QList<Tuple> tuples = Common::Sort(OpenCVUtils::matrixToVector<float>(m));

        float prevValue = 0;
        int prevRank = 0;
        for (int i=0; i<tuples.size(); i++) {
            int rank;
            if (tuples[i].first == prevValue) rank = prevRank;
            else                              rank = i;
            dst.m().at<float>(tuples[i].second / m.cols, tuples[i].second % m.cols) = rank;
            prevValue = tuples[i].first;
            prevRank = rank;
        }
    }
};

BR_REGISTER(Transform, RankTransform)

} // namespace br

#include "rank.moc"
