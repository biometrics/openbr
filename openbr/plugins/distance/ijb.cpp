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

    inline QList<bool> setAlignment(const QList<bool> &list) const
    {
        bool aligned = false;
        foreach (bool a, list)
            if (a) { aligned = true; break; }

        if (aligned) return list; // if some templates are aligned just return the list

        QList<bool> realigned; // if no templates are aligned we treat all templates as aligned
        for (int i = 0; i < list.size(); i++)
            realigned.append(true);

        return realigned;
    }

    inline float compareAverage(const Template &a, const Template &b) const
    {
        QList<bool> aAlign = setAlignment(a.file.getList<bool>("well-aligned"));
        QList<bool> bAlign = setAlignment(b.file.getList<bool>("well-aligned"));

        int count = 0; float score = 0.0f;
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

    inline float compareMax(const Template &a, const Template &b) const
    {
        QList<bool> aAlign = setAlignment(a.file.getList<bool>("well-aligned"));
        QList<bool> bAlign = setAlignment(b.file.getList<bool>("well-aligned"));

        float score = -std::numeric_limits<float>::max();
        for (int i = 0; i < a.size(); i++) {
            if (!aAlign[i]) continue;
            for (int j = 0; j < b.size(); j++) {
                if (!bAlign[j]) continue;
                float tScore = distance->compare(a[i], b[j]);
                if (tScore > score)
                    score = tScore;
            }
        }
        return score;
    }

    inline float compareMin(const Template &a, const Template &b) const
    {
        QList<bool> aAlign = setAlignment(a.file.getList<bool>("well-aligned"));
        QList<bool> bAlign = setAlignment(b.file.getList<bool>("well-aligned"));

        float score = std::numeric_limits<float>::max();
        for (int i = 0; i < a.size(); i++) {
            if (!aAlign[i]) continue;
            for (int j = 0; j < b.size(); j++) {
                if (!bAlign[j]) continue;
                float tScore = distance->compare(a[i], b[j]);
                if (tScore < score)
                    score = tScore;
            }
        }
        return score;
    }
};

BR_REGISTER(Distance, IJBDistance)

} // namespace br

#include "distance/ijb.moc"
