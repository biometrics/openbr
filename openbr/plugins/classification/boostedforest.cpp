#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/boost.h>

using namespace cv;

namespace br
{

struct Node
{
    Node() : left(NULL), right(NULL) {}

<<<<<<< HEAD
    float value;

    float threshold; // For ordered features
    QList<int> subset; // For categorical features
    int featureIdx;

=======
    int featureIdx;
    float threshold; // for ordered features only
    QList<int> subset; // for categorical features only
    float value; // for leaf nodes only
>>>>>>> 4fab7f69ddc82d6ba40a73fc6233e3cc9871473e
    Node *left;
    Node *right;
};

<<<<<<< HEAD
static void buildTreeRecursive(Node *node, const CvDTreeNode *cv_node, int maxCatCount)
{
    if (!cv_node->left) // Write the leaf value
        node->value = cv_node->value; // value of the node. Only relevant for leaf nodes
    else { // Write the splitting information and then the children
        if (maxCatCount > 1)
            for (int i = 0; i < ((maxCatCount + 31) / 32); i++)
                node->subset.append(cv_node->split->subset[i]); // subset to split on (categorical features)
        else
            node->threshold = cv_node->split->ord.c; // threshold to split on (ordered features)

        node->featureIdx = cv_node->split->var_idx; // feature idx of node

        node->left = new Node;
        buildTreeRecursive(node->left, cv_node->left, maxCatCount);
        node->right = new Node;
        buildTreeRecursive(node->right, cv_node->right, maxCatCount);
=======
static void buildTreeRecursive(Node *node, const CvDTreeNode *tree_node, int maxCatCount)
{
    if (tree_node->left) {
        if (maxCatCount > 1) {
            for (int i = 0; i < (maxCatCount + 31)/32; i++)
                node->subset.append(tree_node->split->subset[i]);
        } else {
            node->threshold = tree_node->split->ord.c;
        }

        node->featureIdx = tree_node->split->var_idx;

        node->left = new Node;
        buildTreeRecursive(node->left, tree_node->left, maxCatCount);
        node->right = new Node;
        buildTreeRecursive(node->right, tree_node->right, maxCatCount);
    } else {
        node->value = tree_node->value;
>>>>>>> 4fab7f69ddc82d6ba40a73fc6233e3cc9871473e
    }
}

static void writeRecursive(FileStorage &fs, const Node *node, int maxCatCount)
{
    bool hasChildren = node->left ? true : false;
    fs << "hasChildren" << hasChildren;

    if (!hasChildren) // Write the leaf value
<<<<<<< HEAD
        fs << "value" << node->value; // value of the node. Only relevant for leaf nodes
    else { // Write the splitting information and then the children
        if (maxCatCount > 1) {
            fs << "subset" << "[:";
=======
        fs << "value" << node->value; // value of the node.
    else { // Write the splitting information and then the children
        if (maxCatCount > 1) {
            fs << "subset" << "[";
>>>>>>> 4fab7f69ddc82d6ba40a73fc6233e3cc9871473e
            for (int i = 0; i < ((maxCatCount + 31) / 32); i++)
                fs << node->subset[i]; // subset to split on (categorical features)
            fs << "]";
        } else {
            fs << "threshold" << node->threshold; // threshold to split on (ordered features)
        }

        fs << "feature_idx" << node->featureIdx; // feature idx of node

        fs << "left" << "{"; writeRecursive(fs, node->left, maxCatCount); fs << "}"; // write left child
        fs << "right" << "{"; writeRecursive(fs, node->right, maxCatCount); fs << "}"; // write right child
    }
}

<<<<<<< HEAD
=======
static void readRecursive(const FileNode &fn, Node *node, int maxCatCount)
{
    bool hasChildren = (int)fn["hasChildren"];
    if (!hasChildren) {
        node->value = (float)fn["value"];
    } else {
        if (maxCatCount > 1) {
            FileNode subset_fn = fn["subset"];
            for (FileNodeIterator subset_it = subset_fn.begin(); subset_it != subset_fn.end(); ++subset_it)
                node->subset.append((int)*subset_it);
        } else {
            node->threshold = (float)fn["threshold"];
        }

        node->featureIdx = (int)fn["feature_idx"];

        node->left = new Node;
        readRecursive(fn["left"], node->left, maxCatCount);
        node->right = new Node;
        readRecursive(fn["right"], node->right, maxCatCount);
    }
}

>>>>>>> 4fab7f69ddc82d6ba40a73fc6233e3cc9871473e
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

    QList<Node*> weakClassifiers;
    float threshold;

    void train(const QList<Mat> &images, const QList<float> &labels)
    {
        CascadeBoostParams params(CvBoost::GENTLE, minTAR, maxFAR, trimRate, maxDepth, maxWeakCount);

        FeatureEvaluator featureEvaluator;
        featureEvaluator.init(representation, images.size());

        for (int i = 0; i < images.size(); i++)
            featureEvaluator.setImage(images[i], labels[i], i);

        CascadeBoost boost;
        boost.train(&featureEvaluator, images.size(), 2048, 2048, params);
<<<<<<< HEAD

        threshold = boost.getThreshold();

        foreach (const CvBoostTree *tree, boost.getClassifiers()) {
=======

        // Convert into simpler, cleaner cascade after training
        threshold = boost.getThreshold();

        foreach (const CvBoostTree *tree, boost.getTrees()) {
>>>>>>> 4fab7f69ddc82d6ba40a73fc6233e3cc9871473e
            Node *root = new Node;
            buildTreeRecursive(root, tree->get_root(), representation->maxCatCount());
            weakClassifiers.append(root);
        }
    }

    float classify(const Mat &image) const
    {
<<<<<<< HEAD
        float sum = 0;
        foreach (const Node *root, weakClassifiers) {
            const Node *node = root;

            while (node->left) {
                if (representation->maxCatCount() > 1) {
                    int c = (int)representation->evaluate(image, node->featureIdx);
                    node = (node->subset[c >> 5] & (1 << (c & 31))) ? node->left : node->right;
                } else {
                    float val = representation->evaluate(image, node->featureIdx);
=======
        Mat pp;
        representation->preprocess(image, pp);

        float sum = 0;

        foreach (const Node *node, weakClassifiers) {
            while (node->left) {
                if (representation->maxCatCount() > 1) {
                    int c = (int)representation->evaluate(pp, node->featureIdx);
                    node = (node->subset[c >> 5] & (1 << (c & 31))) ? node->left : node->right;
                } else {
                    double val = representation->evaluate(pp, node->featureIdx);
>>>>>>> 4fab7f69ddc82d6ba40a73fc6233e3cc9871473e
                    node = val < node->threshold ? node->left : node->right;
                }
            }
            sum += node->value;
        }
<<<<<<< HEAD
        return sum < threshold - FLT_EPSILON ? 0.0 : 1.0;
=======

        if (sum < threshold)
            return 0.0f; //-std::abs(sum);
        return 1.0f; //std::abs(sum);
>>>>>>> 4fab7f69ddc82d6ba40a73fc6233e3cc9871473e
    }

    int numFeatures() const
    {
        return representation->numFeatures();
    }

    int maxCatCount() const
    {
        return representation->maxCatCount();
    }

    Size windowSize() const
    {
        return representation->preWindowSize();
    }

    void write(FileStorage &fs) const
    {
<<<<<<< HEAD
        fs << "weakCount" << weakClassifiers.size();
=======
        fs << "numWeak" << weakClassifiers.size();
>>>>>>> 4fab7f69ddc82d6ba40a73fc6233e3cc9871473e
        fs << "stageThreshold" << threshold;
        fs << "weakClassifiers" << "[";
        foreach (const Node *root, weakClassifiers) {
            fs << "{";
            writeRecursive(fs, root, representation->maxCatCount());
            fs << "}";
        }
        fs << "]";
<<<<<<< HEAD
=======
    }

    void read(const FileNode &node)
    {
        weakClassifiers.reserve((int)node["numWeak"]);
        threshold = (float)node["stageThreshold"];

        FileNode weaks_fn = node["weakClassifiers"];
        for (FileNodeIterator weaks_it = weaks_fn.begin(); weaks_it != weaks_fn.end(); ++weaks_it) {
            FileNode weak_fn = *weaks_it;

            Node *root = new Node;
            readRecursive(weak_fn, root, representation->maxCatCount());

            weakClassifiers.append(root);
        }
>>>>>>> 4fab7f69ddc82d6ba40a73fc6233e3cc9871473e
    }
};

BR_REGISTER(Classifier, BoostedForestClassifier)

} // namespace br

#include "classification/boostedforest.moc"
