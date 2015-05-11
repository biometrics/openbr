#ifndef CASCADE_H
#define CASCADE_H

#include <openbr/openbr_plugin.h>
#include <opencv2/imgproc/imgproc.hpp>

#define CC_CASCADE_PARAMS "cascadeParams"
#define CC_STAGE_TYPE     "stageType"
#define CC_FEATURE_TYPE   "featureType"
#define CC_HEIGHT         "height"
#define CC_WIDTH          "width"

#define CC_STAGE_NUM    "stageNum"
#define CC_STAGES       "stages"
#define CC_STAGE_PARAMS "stageParams"

#define CC_BOOST            "BOOST"
#define CC_MAX_DEPTH        "maxDepth"
#define CC_WEAK_COUNT       "maxWeakCount"
#define CC_STAGE_THRESHOLD  "stageThreshold"
#define CC_WEAK_CLASSIFIERS "weakClassifiers"
#define CC_INTERNAL_NODES   "internalNodes"
#define CC_LEAF_VALUES      "leafValues"

#define CC_FEATURES       "features"
#define CC_FEATURE_PARAMS "featureParams"
#define CC_MAX_CAT_COUNT  "maxCatCount"

#define CC_LBP  "LBP"


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
                                   double scaleFactor=1.1,
                                   int minNeighbors=3,
                                   Size minSize=Size(),
                                   Size maxSize=Size());

    void detectMultiScale( const Mat& image,
                                   vector<Rect>& objects,
                                   vector<int>& rejectLevels,
                                   vector<double>& levelWeights,
                                   double scaleFactor=1.1,
                                   int minNeighbors=3,
                                   Size minSize=Size(),
                                   Size maxSize=Size(),
                                   bool outputRejectLevels=false );

    int predict(const Mat &image, double &weight) const;

    class Data
    {
    public:
        struct DTreeNode
        {
            int featureIdx;
            float threshold; // for ordered features only
            int left;
            int right;
        };

        struct DTree
        {
            int nodeCount;
        };

        struct Stage
        {
            int first;
            int ntrees;
            float threshold;
        };

        bool read(const FileNode &node);

        bool isStumpBased;

        int stageType;
        int featureType;
        int ncategories;
        Size origWinSize;

        vector<Stage> stages;
        vector<DTree> classifiers;
        vector<DTreeNode> nodes;
        vector<float> leaves;
        vector<int> subsets;
    };

    Data data;
    Representation *representation;
};

} // namespace br

#endif // CASCADE_H
