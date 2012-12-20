#include <topsurf/descriptor.h>
#include <topsurf/topsurf.h>
#include <mm_plugin.h>

#include "common/opencvutils.h"
#include "common/qtutils.h"
#include "common/resource.h"

using namespace cv;
using namespace mm;
using namespace std;

class TopSurfInitializer : public Initializer
{
    void initialize() const
    {
        Globals.Abbreviations.insert("TopSurf", "Open!TopSurfExtract(40000):TopSurfCompare");
        Globals.Abbreviations.insert("TopSurfM", "Open!TopSurfExtract(1000000):TopSurfCompare");
        Globals.Abbreviations.insert("TopSurfKNN", "Open!TopSurfExtract+TopSurfKNN");
        Globals.Abbreviations.insert("DocumentClassification", "TopSurfKNN");
    }

    void finalize() const {}
};

MM_REGISTER(Initializer, TopSurfInitializer, false)


class TopSurfResourceMaker : public ResourceMaker<TopSurf>
{
    QString file;

public:
    TopSurfResourceMaker(const QString &dictionary)
    {
        file = Globals.SDKPath + "/models/topsurf/dictionary_" + dictionary;
    }

private:
    TopSurf *make() const
    {
        TopSurf *topSurf = new TopSurf(256, 100);
        if (!topSurf->LoadDictionary(qPrintable(file)))
            qFatal("TopSurfResourceMaker::make failed to load dictionary.");
        return topSurf;
    }
};


/****
TopSurfExtract
    Wrapper to TopSurf::ExtractDescriptor()
    B. Thomee, E.M. Bakker, and M.S. Lew, "TOP-SURF: a visual words toolkit",
    in Proceedings of the 18th ACM International Conference on Multimedia, pp. 1473-1476, Firenze, Italy, 2010.
****/
class TopSurfExtract : public UntrainableFeature
{
    Q_OBJECT
    Q_PROPERTY(QString dictionary READ get_dictionary WRITE set_dictionary)
    MM_MEMBER(QString, dictionary)

    Resource<TopSurf> topSurfResource;

public:
    TopSurfExtract() : topSurfResource(new TopSurfResourceMaker("10000")) {}

private:
    void init()
    {
        topSurfResource.setResourceMaker(new TopSurfResourceMaker(dictionary));
    }

    void project(const Template &src, Template &dst) const
    {
        // Compute descriptor (not thread safe)
        TopSurf *topSurf = topSurfResource.acquire();
        TOPSURF_DESCRIPTOR descriptor;
        IplImage iplSrc = src.m();
        if (!topSurf->ExtractDescriptor(iplSrc, descriptor))
            qFatal("TopSurfExtract::project ExtractDescriptor failure.");
        topSurfResource.release(topSurf);

        // Copy descriptor and clean up
        unsigned char *data;
        int length;
        Descriptor2Array(descriptor, data, length);
        Mat m(1, length, CV_8UC1);
        memcpy(m.data, data, length);
        delete data;
        TopSurf::ReleaseDescriptor(descriptor);
        dst = m;
    }

public:
    static QString args()
    {
        return "10000|20000|40000 dictionary = 10000";
    }

    static TopSurfExtract *make(const QStringList &args)
    {
        (void) args;
        return new TopSurfExtract();
    }
};

MM_REGISTER(Feature, TopSurfExtract, true)


class TopSurfHist : public UntrainableFeature
{
    Q_OBJECT
    Q_PROPERTY(int size READ get_size WRITE set_size)
    MM_MEMBER(int, size)

    void project(const Template &src, Template &dst) const
    {
        TOPSURF_DESCRIPTOR td;
        Array2Descriptor(src.m().data, td);

        Mat m(1, size, CV_32FC1);
        m.setTo(0);
        for (int i=0; i<td.count; i++)
            m.at<float>(0, td.visualword[i].identifier % size)++;

        TopSurf::ReleaseDescriptor(td);
        dst = m;
    }

public:
    static QString args()
    {
        return "int size = 10000";
    }

    static TopSurfHist *make(const QStringList &args)
    {
        (void) args;
        return new TopSurfHist();
    }
};

MM_REGISTER(Feature, TopSurfHist, true)


// Wrapper around TopSurf CompareDescriptors
float TopSurfSimilarity(const Mat &a, const Mat &b, bool cosine)
{
    TOPSURF_DESCRIPTOR tda, tdb;
    Array2Descriptor(a.data, tda);
    Array2Descriptor(b.data, tdb);

    float result;
    if (cosine) result = TopSurf::CompareDescriptorsCosine(tda, tdb);
    else        result = TopSurf::CompareDescriptorsAbsolute(tda, tdb);

    TopSurf::ReleaseDescriptor(tda);
    TopSurf::ReleaseDescriptor(tdb);
    return result;
}


/****
TopSurfCompare
    Wrapper to TopSurf_CompareDescriptors()
    B. Thomee, E.M. Bakker, and M.S. Lew, "TOP-SURF: a visual words toolkit",
    in Proceedings of the 18th ACM International Conference on Multimedia, pp. 1473-1476, Firenze, Italy, 2010.
****/
class TopSurfCompare : public ComparerBase
{
    Q_OBJECT
    Q_PROPERTY(bool cosine READ get_cosine WRITE set_cosine)
    MM_MEMBER(bool, cosine)

    float compare(const Mat &a, const Mat &b) const
    {
        return TopSurfSimilarity(a, b, cosine);
    }

public:
    static QString args()
    {
        return "bool cosine = 1";
    }

    static TopSurfCompare *make(const QStringList &args)
    {
        (void) args;
        return new TopSurfCompare();
    }
};

MM_REGISTER(Comparer, TopSurfCompare, true)


/****
TopSurfKNN
    KNN classifier for TopSurf features.
****/
class TopSurfKNN : public Feature
{
    Q_OBJECT
    Q_PROPERTY(int k READ get_k WRITE set_k)
    Q_PROPERTY(bool cosine READ get_cosine WRITE set_cosine)
    MM_MEMBER(int, k)
    MM_MEMBER(bool, cosine)

    TemplateList data;

private:
    void train(const TemplateList &data)
    {
        this->data = data;
    }

    void project(const Template &src, Template &dst) const
    {
        // Compute distance to each descriptor
        QList< QPair<float, int> > distances; // <distance, label>
        distances.reserve(data.size());
        foreach (const Template &t, data)
            distances.append(QPair<float, int>(TopSurfSimilarity(src, t, cosine), t.file.label()));

        // Find nearest neighbors
        qSort(distances);
        QHash<int, QPair<int, float> > counts; // <label, <count, cumulative distance>>
        for (int i=0; i<k; i++) {
            QPair<float,int> &distance = distances[i];
            QPair<int,float> &count = counts[distance.second];
            count.first++;
            count.second += distance.first;
        }

        // Find most occuring label
        int best_label = -1;
        int best_count = 0;
        float best_distance = numeric_limits<float>::max();
        foreach (int label, counts.keys()) {
            const QPair<int, float> &count = counts[label];
            if ((count.first > best_count) || ((count.first == best_count) && (count.second < best_distance))) {
                best_label = label;
                best_count = count.first;
                best_distance = count.second;
            }
        }
        assert(best_label != -1);

        // Measure confidence
        int rest_count = 0;
        float rest_distance = 0;
        foreach (int label, counts.keys()) {
            if (label != best_label) {
                const QPair<int, float> &count = counts[label];
                rest_count = count.first;
                rest_distance = count.second;
            }
        }

        dst = src;
        dst.file["Label"] = best_label;
        dst.file["Confidence"] = (float)best_count/(float)k;
    }

    void store(QDataStream &stream) const
    {
        stream << data;
    }

    void load(QDataStream &stream)
    {
        stream >> data;
    }

public:
    static QString args()
    {
        return "int k, int cosine = 1";
    }

    static TopSurfKNN *make(const QStringList &args)
    {
        (void) args;
        return new TopSurfKNN();
    }
};

MM_REGISTER(Feature, TopSurfKNN, true)

#include "topsurf.moc"
