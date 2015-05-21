#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/boost.h>

#define THRESHOLD_EPS 1e-5

using namespace cv;

namespace br
{

struct Node
{
    float value; // for leaf nodes

    float threshold; // for ordered features
    QList<int> subset; // for categorical features
    int featureIdx;

    Node *left, *right;
};

static void buildTreeRecursive(Node *node, const CvDTreeNode *cv_node, int maxCatCount)
{
    if (!cv_node->left) {
        node->value = cv_node->value;
        node->left = node->right = NULL;
    } else {
        if (maxCatCount > 0)
            for (int i = 0; i < (maxCatCount + 31)/32; i++)
                node->subset.append(cv_node->split->subset[i]);
        else
            node->threshold = cv_node->split->ord.c;

        node->featureIdx = cv_node->split->var_idx;

        node->left = new Node; node->right = new Node;
        buildTreeRecursive(node->left, cv_node->left, maxCatCount);
        buildTreeRecursive(node->right, cv_node->right, maxCatCount);
    }
}

static void readRecursive(const FileNode &fn, Node *node, int maxCatCount)
{
    bool hasChildren = (int)fn["hasChildren"];
    if (!hasChildren) {
        node->value = (float)fn["value"];
        node->left = node->right = NULL;
    } else {
        if (maxCatCount > 0) {
            FileNode subset_fn = fn["subset"];
            for (FileNodeIterator subset_it = subset_fn.begin(); subset_it != subset_fn.end(); ++subset_it)
                node->subset.append((int)*subset_it);
        } else {
            node->threshold = (float)fn["threshold"];
        }

        node->featureIdx = (int)fn["featureIdx"];

        node->left = new Node; node->right = new Node;
        readRecursive(fn["left"], node->left, maxCatCount);
        readRecursive(fn["right"], node->right, maxCatCount);
    }
}

static void writeRecursive(FileStorage &fs, const Node *node, int maxCatCount)
{
    bool hasChildren = node->left ? true : false;
    fs << "hasChildren" << hasChildren;

    if (!hasChildren)
        fs << "value" << node->value;
    else {
        if (maxCatCount > 0) {
            fs << "subset" << "[";
            for (int i = 0; i < (maxCatCount + 31)/32; i++)
                fs << node->subset[i];
            fs << "]";
        } else {
            fs << "threshold" << node->threshold;
        }

        fs << "featureIdx" << node->featureIdx;

        fs << "left" << "{"; writeRecursive(fs, node->left, maxCatCount); fs << "}";
        fs << "right" << "{"; writeRecursive(fs, node->right, maxCatCount); fs << "}";
    }
}

class BoostedForestClassifier : public Classifier
{
    Q_OBJECT

    Q_PROPERTY(br::Representation *representation READ get_representation WRITE set_representation RESET reset_representation STORED false)
    Q_PROPERTY(float minTAR READ get_minTAR WRITE set_minTAR RESET reset_minTAR STORED false)
    Q_PROPERTY(float maxFAR READ get_maxFAR WRITE set_maxFAR RESET reset_maxFAR STORED false)
    Q_PROPERTY(float trimRate READ get_trimRate WRITE set_trimRate RESET reset_trimRate STORED false)
    Q_PROPERTY(int maxDepth READ get_maxDepth WRITE set_maxDepth RESET reset_maxDepth STORED false)
    Q_PROPERTY(int maxWeakCount READ get_maxWeakCount WRITE set_maxWeakCount RESET reset_maxWeakCount STORED false)

    BR_PROPERTY(br::Representation *, representation, NULL)
    BR_PROPERTY(float, minTAR, 0.995)
    BR_PROPERTY(float, maxFAR, 0.5)
    BR_PROPERTY(float, trimRate, 0.95)
    BR_PROPERTY(int, maxDepth, 1)
    BR_PROPERTY(int, maxWeakCount, 100)

    QList<Node*> classifiers;
    float threshold;

    void train(const QList<Mat> &images, const QList<float> &labels)
    {
        CascadeBoostParams params(CvBoost::GENTLE, minTAR, maxFAR, trimRate, maxDepth, maxWeakCount);

        FeatureEvaluator featureEvaluator;
        featureEvaluator.init(representation, images.size());

        for (int i = 0; i < images.size(); i++)
            featureEvaluator.setImage(images[i], labels[i], i);

        CascadeBoost boost;
        boost.train(&featureEvaluator, images.size(), 1024, 1024, params);

        threshold = boost.getThreshold();

        foreach (const CvBoostTree *classifier, boost.getClassifers()) {
            Node *root = new Node;
            buildTreeRecursive(root, classifier->get_root(), representation->maxCatCount());
            classifiers.append(root);
        }
    }

    float classify(const Mat &image, float &confidence) const
    {
        confidence = 0;
        for (int i = 0; i < classifiers.size(); i++) {
            Node *node = classifiers[i];

            while (node->left) {
                if (representation->maxCatCount() > 1) {
                    int c = (int)representation->evaluate(image, node->featureIdx);
                    node = (node->subset[c >> 5] & (1 << (c & 31))) ? node->left : node->right;
                } else {
                    double val = representation->evaluate(image, node->featureIdx);
                    node = val <= node->threshold ? node->left : node->right;
                }
            }
            confidence += node->value;
        }

        return confidence < threshold - THRESHOLD_EPS ? 0.0f : 1.0f;
    }

    int numFeatures() const
    {
        return representation->numFeatures();
    }

    int maxCatCount() const
    {
        return representation->maxCatCount();
    }

    Mat preprocess(const Mat &image) const
    {
        Mat dst;
        representation->preprocess(image, dst);
        return dst;
    }

    Size windowSize() const
    {
        return representation->preWindowSize();
    }

    void read(const FileNode &node)
    {
        threshold = (float)node["stageThreshold"];
        FileNode weaks_fn = node["weakClassifiers"];
        for (FileNodeIterator weaks_it = weaks_fn.begin(); weaks_it != weaks_fn.end(); ++weaks_it) {
            Node *root = new Node;
            readRecursive(*weaks_it, root, representation->maxCatCount());
            classifiers.append(root);
        }
    }

    void write(FileStorage &fs) const
    {
        fs << "stageThreshold" << threshold;
        fs << "weakSize" << classifiers.size();
        fs << "weakClassifiers" << "[";
        foreach (const Node *root, classifiers) {
            fs << "{";
            writeRecursive(fs, root, representation->maxCatCount());
            fs << "}";
        }
        fs << "]";
    }
};

BR_REGISTER(Classifier, BoostedForestClassifier)

} // namespace br

#include "classification/boostedforest.moc"
