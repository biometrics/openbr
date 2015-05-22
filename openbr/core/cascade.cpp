#include "cascade.h"

using namespace br;

void br::groupRectangles(vector<Rect>& rectList, int groupThreshold, double eps, vector<int>* weights, vector<double>* levelWeights)
{
    if( groupThreshold <= 0 || rectList.empty() )
    {
        if( weights )
        {
            size_t i, sz = rectList.size();
            weights->resize(sz);
            for( i = 0; i < sz; i++ )
                (*weights)[i] = 1;
        }
        return;
    }

    vector<int> labels;
    int nclasses = partition(rectList, labels, SimilarRects(eps));

    vector<Rect> rrects(nclasses);
    vector<int> rweights(nclasses, 0);
    vector<int> rejectLevels(nclasses, 0);
    vector<double> rejectWeights(nclasses, DBL_MIN);
    int i, j, nlabels = (int)labels.size();
    for( i = 0; i < nlabels; i++ )
    {
        int cls = labels[i];
        rrects[cls].x += rectList[i].x;
        rrects[cls].y += rectList[i].y;
        rrects[cls].width += rectList[i].width;
        rrects[cls].height += rectList[i].height;
        rweights[cls]++;
    }
    if ( levelWeights && weights && !weights->empty() && !levelWeights->empty() )
    {
        for( i = 0; i < nlabels; i++ )
        {
            int cls = labels[i];
            if( (*weights)[i] > rejectLevels[cls] )
            {
                rejectLevels[cls] = (*weights)[i];
                rejectWeights[cls] = (*levelWeights)[i];
            }
            else if( ( (*weights)[i] == rejectLevels[cls] ) && ( (*levelWeights)[i] > rejectWeights[cls] ) )
                rejectWeights[cls] = (*levelWeights)[i];
        }
    }

    for( i = 0; i < nclasses; i++ )
    {
        Rect r = rrects[i];
        float s = 1.f/rweights[i];
        rrects[i] = Rect(saturate_cast<int>(r.x*s),
             saturate_cast<int>(r.y*s),
             saturate_cast<int>(r.width*s),
             saturate_cast<int>(r.height*s));
    }

    rectList.clear();
    if( weights )
        weights->clear();
    if( levelWeights )
        levelWeights->clear();

    for( i = 0; i < nclasses; i++ )
    {
        Rect r1 = rrects[i];
        int n1 = levelWeights ? rejectLevels[i] : rweights[i];

        double w1 = rejectWeights[i];
        if( n1 <= groupThreshold )
            continue;
        // filter out small face rectangles inside large rectangles
        for( j = 0; j < nclasses; j++ )
        {
            int n2 = rweights[j];

            if( j == i || n2 <= groupThreshold )
                continue;
            Rect r2 = rrects[j];

            int dx = saturate_cast<int>( r2.width * eps );
            int dy = saturate_cast<int>( r2.height * eps );

            if( i != j &&
                r1.x >= r2.x - dx &&
                r1.y >= r2.y - dy &&
                r1.x + r1.width <= r2.x + r2.width + dx &&
                r1.y + r1.height <= r2.y + r2.height + dy &&
                (n2 > std::max(3, n1) || n1 < 3) )
                break;
        }

        if( j == nclasses )
        {
            rectList.push_back(r1);
            if( weights )
                weights->push_back(n1);
            if( levelWeights )
                levelWeights->push_back(w1);
        }
    }
}

void br::groupRectangles(vector<Rect>& rectList, int groupThreshold, double eps)
{
    groupRectangles(rectList, groupThreshold, eps, 0, 0);
}

void br::groupRectangles(vector<Rect>& rectList, vector<int>& weights, int groupThreshold, double eps)
{
    groupRectangles(rectList, groupThreshold, eps, &weights, 0);
}

void br::groupRectangles(vector<Rect>& rectList, vector<int>& rejectLevels, vector<double>& levelWeights, int groupThreshold, double eps)
{
    groupRectangles(rectList, groupThreshold, eps, &rejectLevels, &levelWeights);
}

// --------------------------------- Cascade Classifier ----------------------------------

static void loadRecursive(const FileNode &fn, _CascadeClassifier::Node *node, int maxCatCount)
{
    bool hasChildren = (int)fn["hasChildren"];
    if (!hasChildren)
        node->value = (float)fn["value"];
    else {
        if (maxCatCount > 0) {
            FileNode subset_fn = fn["subset"];
            for (FileNodeIterator subset_it = subset_fn.begin(); subset_it != subset_fn.end(); ++subset_it)
                node->subset.append((int)*subset_it);
        } else {
            node->threshold = (float)fn["threshold"];
        }

        node->featureIdx = (int)fn["featureIdx"];

        node->left = new _CascadeClassifier::Node; node->right = new _CascadeClassifier::Node;
        loadRecursive(fn["left"], node->left, maxCatCount);
        loadRecursive(fn["right"], node->right, maxCatCount);
    }
}

bool _CascadeClassifier::load(const string& filename)
{
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
        return false;

    FileNode root = fs.getFirstTopLevelNode();

    const float THRESHOLD_EPS = 1e-5;

    int maxCatCount = representation->maxCatCount();

    // load stages
    FileNode stages_fn = root["stages"];
    if( stages_fn.empty() )
        return false;

    for (FileNodeIterator stage_it = stages_fn.begin(); stage_it != stages_fn.end(); ++stage_it) {
        FileNode stage_fn = *stage_it;

        Stage stage;
        stage.threshold = (float)stage_fn["stageThreshold"] - THRESHOLD_EPS;

        FileNode nodes_fn = stage_fn["weakClassifiers"];
        if(nodes_fn.empty())
            return false;

        for (FileNodeIterator node_it = nodes_fn.begin(); node_it != nodes_fn.end(); ++node_it) {
            FileNode node_fn = *node_it;

            Node *root = new Node;
            loadRecursive(node_fn, root, maxCatCount);

            stage.trees.append(root);
        }

        stages.append(stage);
    }

    return true;
}

int _CascadeClassifier::predict(const Mat &image, double &sum) const
{
    for (int stageIdx = 0; stageIdx < stages.size(); stageIdx++) {
        Stage stage = stages[stageIdx];
        sum = 0;

        for (int treeIdx = 0; treeIdx < stage.trees.size(); treeIdx++) {
            Node *node = stage.trees[treeIdx];

            while (node->left) {
                if (representation->maxCatCount() > 1) {
                    int c = (int)representation->evaluate(image, node->featureIdx);
                    node = (node->subset[c >> 5] & (1 << (c & 31))) ? node->left : node->right;
                } else {
                    double val = representation->evaluate(image, node->featureIdx);
                    node = val < node->threshold ? node->left : node->right;
                }
            }
            sum += node->value;
        }

        if (sum < stage.threshold)
            return stageIdx;
    }

    return stages.size();
}

void _CascadeClassifier::detectMultiScale(const Mat& image, vector<Rect>& objects, vector<int>& rejectLevels,
                                                            vector<double>& levelWeights,
                                                            double scaleFactor, int minNeighbors,
                                                            Size minSize, Size maxSize) const
{
    const double GROUP_EPS = 0.2;

    CV_Assert( scaleFactor > 1 && image.depth() == CV_8U );

    if (stages.empty())
        return;

    if( maxSize.height == 0 || maxSize.width == 0 )
        maxSize = image.size();

    Mat imageBuffer(image.rows + 1, image.cols + 1, CV_8U);

    for (double factor = 1; ; factor *= scaleFactor) {
        int dx, dy;
        Size originalWindowSize = representation->windowSize(&dx, &dy);

        Size windowSize(cvRound(originalWindowSize.width*factor), cvRound(originalWindowSize.height*factor) );
        Size scaledImageSize(cvRound(image.cols/factor ), cvRound(image.rows/factor));
        Size processingRectSize(scaledImageSize.width - originalWindowSize.width, scaledImageSize.height - originalWindowSize.height);

        if (processingRectSize.width <= 0 || processingRectSize.height <= 0)
            break;
        if (windowSize.width > maxSize.width || windowSize.height > maxSize.height)
            break;
        if (windowSize.width < minSize.width || windowSize.height < minSize.height)
            continue;

        Mat scaledImage(scaledImageSize, CV_8U, imageBuffer.data);
        resize(image, scaledImage, scaledImageSize, 0, 0, CV_INTER_LINEAR);

        Mat repImage;
        representation->preprocess(scaledImage, repImage);

        int yStep = factor > 2. ? 1 : 2;
        for (int y = 0; y < processingRectSize.height; y += yStep) {
            for (int x = 0; x < processingRectSize.width; x += yStep) {
                Mat window = repImage(Rect(Point(x, y), Size(originalWindowSize.width + dx, originalWindowSize.height + dy))).clone();

                double gypWeight;
                int result = predict(window, gypWeight);

                if (stages.size() - result < 4) {
                    objects.push_back(Rect(cvRound(x*factor), cvRound(y*factor), windowSize.width, windowSize.height));
                    rejectLevels.push_back(result);
                    levelWeights.push_back(gypWeight);
                }

                if (result == 0)
                    x += yStep;
            }
        }
    }

    groupRectangles(objects, rejectLevels, levelWeights, minNeighbors, GROUP_EPS);
}
