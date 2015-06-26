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

static void loadRecursive(QDataStream &stream, Node *node, int maxCatCount)
{
    bool hasChildren; stream >> hasChildren;

    if (!hasChildren) {
        stream >> node->value;
        node->left = node->right = NULL;
    } else {
        if (maxCatCount > 0)
            for (int i = 0; i < (maxCatCount + 31)/32; i++) {
                int s; stream >> s; node->subset.append(s);
            }
        else
            stream >> node->threshold;

        stream >> node->featureIdx;

        node->left = new Node; node->right = new Node;
        loadRecursive(stream, node->left, maxCatCount);
        loadRecursive(stream, node->right, maxCatCount);
    }
}

static void storeRecursive(QDataStream &stream, const Node *node, int maxCatCount)
{
    bool hasChildren = node->left ? true : false;
    stream << hasChildren;

    if (!hasChildren)
        stream << node->value;
    else {
        if (maxCatCount > 0)
            for (int i = 0; i < (maxCatCount + 31)/32; i++)
                stream << node->subset[i];
        else
            stream << node->threshold;

        stream << node->featureIdx;

        storeRecursive(stream, node->left, maxCatCount);
        storeRecursive(stream, node->right, maxCatCount);
    }
}

/*!
 * \brief A classification wrapper on OpenCV's CvBoost class. It uses CvBoost for training a boosted forest and then performs classification using the trained nodes.
 * \author Jordan Cheney \cite jcheney
 * \author Scott Klum \cite sklum
 * \br_property Representation* representation The Representation describing the features used by the boosted forest
 * \br_property float minTAR The minimum true accept rate during training
 * \br_property float maxFAR The maximum false accept rate during training
 * \br_property float trimRate The trim rate during training
 * \br_property int maxDepth The maximum depth for each trained tree
 * \br_property int maxWeakCount The maximum number of trees in the forest
 * \br_property Type type. The type of boosting to perform. Options are [Discrete, Real, Logit, Gentle]. Gentle is the default.
 */
class BoostedForestClassifier : public Classifier
{
    Q_OBJECT
    Q_ENUMS(Type)

    Q_PROPERTY(br::Representation* representation READ get_representation WRITE set_representation RESET reset_representation STORED false)
    Q_PROPERTY(float minTAR READ get_minTAR WRITE set_minTAR RESET reset_minTAR STORED false)
    Q_PROPERTY(float maxFAR READ get_maxFAR WRITE set_maxFAR RESET reset_maxFAR STORED false)
    Q_PROPERTY(float trimRate READ get_trimRate WRITE set_trimRate RESET reset_trimRate STORED false)
    Q_PROPERTY(int maxDepth READ get_maxDepth WRITE set_maxDepth RESET reset_maxDepth STORED false)
    Q_PROPERTY(int maxWeakCount READ get_maxWeakCount WRITE set_maxWeakCount RESET reset_maxWeakCount STORED false)
    Q_PROPERTY(Type type READ get_type WRITE set_type RESET reset_type STORED false)

public:
    enum Type { Discrete = CvBoost::DISCRETE,
                Real = CvBoost::REAL,
                Logit = CvBoost::LOGIT,
                Gentle = CvBoost::GENTLE};
private:
    BR_PROPERTY(br::Representation*, representation, NULL)
    BR_PROPERTY(float, minTAR, 0.995)
    BR_PROPERTY(float, maxFAR, 0.5)
    BR_PROPERTY(float, trimRate, 0.95)
    BR_PROPERTY(int, maxDepth, 1)
    BR_PROPERTY(int, maxWeakCount, 100)
    BR_PROPERTY(Type, type, Gentle)

    QList<Node*> classifiers;
    float threshold;

    void train(const QList<Mat> &images, const QList<float> &labels)
    {
        representation->train(images, labels);

        CascadeBoostParams params(type, minTAR, maxFAR, trimRate, maxDepth, maxWeakCount);

        FeatureEvaluator featureEvaluator;
        featureEvaluator.init(representation, images.size(), representation->numChannels());

        for (int i = 0; i < images.size(); i++)
            featureEvaluator.setImage(images[i], labels[i], i);

        CascadeBoost boost;
        boost.train(&featureEvaluator, images.size(), 1024, 1024, representation->numChannels(), params);

        threshold = boost.getThreshold();

        foreach (const CvBoostTree *classifier, boost.getClassifers()) {
            Node *root = new Node;
            buildTreeRecursive(root, classifier->get_root(), representation->maxCatCount());
            classifiers.append(root);
        }
    }

    float classify(const Mat &image, bool process, float *confidence) const
    {
        Mat m;
        if (process)
            m = preprocess(image);
        else
            m = image;

        float sum = 0;
        for (int i = 0; i < classifiers.size(); i++) {
            Node *node = classifiers[i];

            while (node->left) {
                if (representation->maxCatCount() > 0) {
                    int c = (int)representation->evaluate(m, node->featureIdx);
                    node = (node->subset[c >> 5] & (1 << (c & 31))) ? node->left : node->right;
                } else {
                    double val = representation->evaluate(m, node->featureIdx);
                    node = val <= node->threshold ? node->left : node->right;
                }
            }

            sum += node->value;
        }

        if (confidence)
            *confidence = sum;
        return sum < threshold - THRESHOLD_EPS ? 0.0f : 1.0f;
    }

    int numFeatures() const
    {
        return representation->numFeatures();
    }

    Mat preprocess(const Mat &image) const
    {
        Mat dst;
        representation->preprocess(image, dst);
        return dst;
    }

    Size windowSize(int *dx, int *dy) const
    {
        return representation->windowSize(dx, dy);
    }

    void load(QDataStream &stream)
    {
        representation->load(stream);

        stream >> threshold;
        int numClassifiers; stream >> numClassifiers;
        for (int i = 0; i < numClassifiers; i++) {
            Node *classifier = new Node;
            loadRecursive(stream, classifier, representation->maxCatCount());
            classifiers.append(classifier);
        }
    }

    void store(QDataStream &stream) const
    {
        representation->store(stream);

        stream << threshold;
        stream << classifiers.size();
        foreach (const Node *classifier, classifiers)
            storeRecursive(stream, classifier, representation->maxCatCount());
    }
};

BR_REGISTER(Classifier, BoostedForestClassifier)

} // namespace br

#include "classification/boostedforest.moc"
