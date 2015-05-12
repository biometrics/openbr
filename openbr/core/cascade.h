#ifndef CASCADE_H
#define CASCADE_H

#include <openbr/openbr_plugin.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

namespace br
{

// class for grouping object candidates, detected by Cascade Classifier, HOG etc.
// instance of the class is to be passed to cv::partition (see cxoperations.hpp)
class SimilarRects
{
public:
    SimilarRects(double _eps) : eps(_eps) {}
    inline bool operator()(const Rect& r1, const Rect& r2) const
    {
        double delta = eps*(std::min(r1.width, r2.width) + std::min(r1.height, r2.height))*0.5;
        return std::abs(r1.x - r2.x) <= delta &&
            std::abs(r1.y - r2.y) <= delta &&
            std::abs(r1.x + r1.width - r2.x - r2.width) <= delta &&
            std::abs(r1.y + r1.height - r2.y - r2.height) <= delta;
    }
    double eps;
};

void groupRectangles(vector<Rect>& rectList, int groupThreshold, double eps=0.2);
void groupRectangles(vector<Rect>& rectList, vector<int>& weights, int groupThreshold, double eps=0.2);
void groupRectangles(vector<Rect>& rectList, int groupThreshold, double eps, vector<int>* weights, vector<double>* levelWeights );
void groupRectangles(vector<Rect>& rectList, vector<int>& rejectLevels, vector<double>& levelWeights, int groupThreshold, double eps=0.2);

class _CascadeClassifier
{
public:
    _CascadeClassifier() : representation(Representation::make("MBLBP(24,24)", NULL)) {}
    _CascadeClassifier(const string& filename) : representation(Representation::make("MBLBP(24,24)", NULL)) { load(filename); }
    ~_CascadeClassifier() {}

    bool load(const string& filename);

    void detectMultiScale(const Mat& image,
                                   vector<Rect>& objects,
                                   vector<int>& rejectLevels,
                                   vector<double>& levelWeights,
                                   double scaleFactor=1.1,
                                   int minNeighbors=3,
                                   Size minSize=Size(),
                                   Size maxSize=Size()) const;

    int predict(const Mat &image, double &weight) const;

    struct Node
    {
        Node() : left(NULL), right(NULL) {}

        int featureIdx;
        float threshold; // for ordered features only
        QList<int> subset; // for categorical features only
        float value; // for leaf nodes only
        Node *left;
        Node *right;
    };

    struct Stage
    {
        QList<Node*> trees;
        float threshold;
    };

    QList<Stage> stages;
    Representation *representation;
};

} // namespace br

#endif // CASCADE_H
