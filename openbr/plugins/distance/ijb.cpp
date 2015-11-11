#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

class IJBDistance : public UntrainableDistance
{
    Q_OBJECT

    Q_ENUMS(CompareMethod);

public:
    enum CompareMethod { AVERAGE, MAX, MIN };

private:
    Q_PROPERTY(br::Distance* distance READ get_distance WRITE set_distance RESET reset_distance STORED false)
    Q_PROPERTY(CompareMethod method READ get_method WRITE set_method RESET reset_method STORED false)
    BR_PROPERTY(br::Distance*, distance, NULL)
    BR_PROPERTY(CompareMethod, method, AVERAGE)

    float compare(const Template &a, const Template &b) const
    {
        if (method == AVERAGE)  return compareAverage(a, b);
        else if (method == MAX) return compareMax(a, b);
        else if (method == MIN) return compareMin(a, b);

        return -std::numeric_limits<float>::max();
    }

    inline float compareAverage(const Template &a, const Template &b) const
    {
        float score = 0.0f;
        QList<bool> aAlign = a.file.getList<bool>("well-aligned");
        QList<bool> bAlign = b.file.getList<bool>("well-aligned");

        bool alignedA = false, alignedB = false;
        foreach (bool ab, aAlign)
            if (ab) alignedA = true;
        foreach (bool bb, bAlign)
            if (bb) alignedB = true;

        if (alignedA && alignedB) {
            int count = 0;
            for (int i = 0; i < a.size(); i++) {
                if (!aAlign[i]) continue;
                for (int j = 0; j < b.size(); j++) {
                    if (!bAlign[j]) continue;
                    score += distance->compare(a[i], b[j]);
                    count++;
                }
            }
            return score / count;
        }

        foreach (const Mat &ma, a)
            foreach (const Mat &mb, b)
                score += distance->compare(ma, mb);
        return score / (a.size() + b.size());
    }

    inline float compareMax(const Template &a, const Template &b) const
    {
        float score = -std::numeric_limits<float>::max();
        foreach (const Mat &ma, a) {
            foreach (const Mat &mb, b) {
                float dist = distance->compare(ma, mb);
                if (dist > score)
                    score = dist;
            }
        }
        return score;
    }

    inline float compareMin(const Template &a, const Template &b) const
    {
        float score = std::numeric_limits<float>::max();
        foreach (const Mat &ma, a) {
            foreach (const Mat &mb, b) {
                float dist = distance->compare(ma, mb);
                if (dist < score)
                    score = dist;
            }
        }
        return score;
    }
};

BR_REGISTER(Distance, IJBDistance)

} // namespace br

#include "distance/ijb.moc"
