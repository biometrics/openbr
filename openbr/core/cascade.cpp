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

bool _CascadeClassifier::load(const string& filename)
{
    data = Data();

    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
        return false;

    return data.read(fs.getFirstTopLevelNode());
}

int _CascadeClassifier::predict(const Mat &image, double &sum) const
{
    int nstages = (int)data.stages.size();
    int nodeOfs = 0, leafOfs = 0;

    size_t subsetSize = (data.ncategories + 31)/32;
    const int *cascadeSubsets = &data.subsets[0];

    const float *cascadeLeaves = &data.leaves[0];
    const Data::DTreeNode *cascadeNodes = &data.nodes[0];
    const Data::DTree *cascadeWeaks = &data.classifiers[0];
    const Data::Stage *cascadeStages = &data.stages[0];

    for (int stageIdx = 0; stageIdx < nstages; stageIdx++) {
        const Data::Stage &stage = cascadeStages[stageIdx];
        sum = 0;

        for (int wi = 0; wi < stage.ntrees; wi++) {
            const Data::DTree &weak = cascadeWeaks[stage.first + wi];
            int idx = 0, root = nodeOfs;

            do {
                const Data::DTreeNode &node = cascadeNodes[root + idx];
                if (data.ncategories > 0) {
                    int c = (int)representation->evaluate(image, node.featureIdx);
                    const int* subset = &cascadeSubsets[(root + idx)*subsetSize];
                    idx = (subset[c>>5] & (1 << (c & 31))) ? node.left : node.right;
                } else {
                    double val = representation->evaluate(image, node.featureIdx);
                    idx = val < node.threshold ? node.left : node.right;
                }
            } while( idx > 0 );

            sum += cascadeLeaves[leafOfs - idx];
            nodeOfs += weak.nodeCount;
            leafOfs += weak.nodeCount + 1;
        }
        if( sum < stage.threshold )
            return -stageIdx;
    }
    return 1;
}

void _CascadeClassifier::detectMultiScale( const Mat& image, vector<Rect>& objects,
                                          vector<int>& rejectLevels,
                                          vector<double>& levelWeights,
                                          double scaleFactor, int minNeighbors,
                                          Size minObjectSize, Size maxObjectSize,
                                          bool outputRejectLevels )
{
    const double GROUP_EPS = 0.2;

    CV_Assert( scaleFactor > 1 && image.depth() == CV_8U );

    if (data.stages.empty())
        return;

    if( maxObjectSize.height == 0 || maxObjectSize.width == 0 )
        maxObjectSize = image.size();

    Mat imageBuffer(image.rows + 1, image.cols + 1, CV_8U);

    for (double factor = 1; ; factor *= scaleFactor) {
        Size originalWindowSize = data.origWinSize;

        Size windowSize(cvRound(originalWindowSize.width*factor), cvRound(originalWindowSize.height*factor) );
        Size scaledImageSize(cvRound(image.cols/factor ), cvRound(image.rows/factor));
        Size processingRectSize(scaledImageSize.width - originalWindowSize.width, scaledImageSize.height - originalWindowSize.height);

        if (processingRectSize.width <= 0 || processingRectSize.height <= 0)
            break;
        if (windowSize.width > maxObjectSize.width || windowSize.height > maxObjectSize.height)
            break;
        if (windowSize.width < minObjectSize.width || windowSize.height < minObjectSize.height)
            continue;

        Mat scaledImage(scaledImageSize, CV_8U, imageBuffer.data);
        resize(image, scaledImage, scaledImageSize, 0, 0, CV_INTER_LINEAR);

        Mat repImage;
        representation->preprocess(scaledImage, repImage);

        int yStep = factor > 2. ? 1 : 2;
        for (int y = 0; y < processingRectSize.height; y += yStep) {
            for (int x = 0; x < processingRectSize.width; x += yStep) {
                Mat window = repImage(Rect(Point(x, y), representation->postWindowSize())).clone();

                double gypWeight;
                int result = predict(window, gypWeight);

                if (outputRejectLevels) {
                    if (result == 1)
                        result = -(int)data.stages.size();
                    if (data.stages.size() + result < 4) {
                        objects.push_back(Rect(cvRound(x*factor), cvRound(y*factor), windowSize.width, windowSize.height));
                        rejectLevels.push_back(-result);
                        levelWeights.push_back(gypWeight);
                    }
                }
                else if (result > 0) {
                    objects.push_back(Rect(cvRound(x*factor), cvRound(y*factor), windowSize.width, windowSize.height));
                }
                if (result == 0)
                    x += yStep;
            }
        }
    }

    if (outputRejectLevels)
        groupRectangles(objects, rejectLevels, levelWeights, minNeighbors, GROUP_EPS);
    else
        groupRectangles(objects, minNeighbors, GROUP_EPS);
}

void _CascadeClassifier::detectMultiScale(const Mat& image, vector<Rect>& objects,
                                          double scaleFactor, int minNeighbors, Size minObjectSize, Size maxObjectSize)
{
    vector<int> fakeLevels;
    vector<double> fakeWeights;
    detectMultiScale( image, objects, fakeLevels, fakeWeights, scaleFactor,
        minNeighbors, minObjectSize, maxObjectSize, false );
}

bool _CascadeClassifier::Data::read(const FileNode &root)
{
    static const float THRESHOLD_EPS = 1e-5f;

    origWinSize.width = (int)root[CC_WIDTH];
    origWinSize.height = (int)root[CC_HEIGHT];
    CV_Assert( origWinSize.height > 0 && origWinSize.width > 0 );

    isStumpBased = (int)(root[CC_STAGE_PARAMS][CC_MAX_DEPTH]) == 1 ? true : false;

    // load feature params
    FileNode fn = root[CC_FEATURE_PARAMS];
    if( fn.empty() )
        return false;

    ncategories = fn[CC_MAX_CAT_COUNT];
    int subsetSize = (ncategories + 31)/32,
        nodeStep = 3 + ( ncategories>0 ? subsetSize : 1 );

    // load stages
    fn = root[CC_STAGES];
    if( fn.empty() )
        return false;

    stages.reserve(fn.size());
    classifiers.clear();
    nodes.clear();

    FileNodeIterator it = fn.begin(), it_end = fn.end();

    for( int si = 0; it != it_end; si++, ++it )
    {
        FileNode fns = *it;
        Stage stage;
        stage.threshold = (float)fns[CC_STAGE_THRESHOLD] - THRESHOLD_EPS;
        fns = fns[CC_WEAK_CLASSIFIERS];
        if(fns.empty())
            return false;
        stage.ntrees = (int)fns.size();
        stage.first = (int)classifiers.size();
        stages.push_back(stage);
        classifiers.reserve(stages[si].first + stages[si].ntrees);

        FileNodeIterator it1 = fns.begin(), it1_end = fns.end();
        for( ; it1 != it1_end; ++it1 ) // weak trees
        {
            FileNode fnw = *it1;
            FileNode internalNodes = fnw[CC_INTERNAL_NODES];
            FileNode leafValues = fnw[CC_LEAF_VALUES];
            if( internalNodes.empty() || leafValues.empty() )
                return false;

            DTree tree;
            tree.nodeCount = (int)internalNodes.size()/nodeStep;
            classifiers.push_back(tree);

            nodes.reserve(nodes.size() + tree.nodeCount);
            leaves.reserve(leaves.size() + leafValues.size());
            if( subsetSize > 0 )
                subsets.reserve(subsets.size() + tree.nodeCount*subsetSize);

            FileNodeIterator internalNodesIter = internalNodes.begin(), internalNodesEnd = internalNodes.end();

            for( ; internalNodesIter != internalNodesEnd; ) // nodes
            {
                DTreeNode node;
                node.left = (int)*internalNodesIter; ++internalNodesIter;
                node.right = (int)*internalNodesIter; ++internalNodesIter;
                node.featureIdx = (int)*internalNodesIter; ++internalNodesIter;
                if( subsetSize > 0 )
                {
                    for( int j = 0; j < subsetSize; j++, ++internalNodesIter )
                        subsets.push_back((int)*internalNodesIter);
                    node.threshold = 0.f;
                }
                else
                {
                    node.threshold = (float)*internalNodesIter; ++internalNodesIter;
                }
                nodes.push_back(node);
            }

            internalNodesIter = leafValues.begin(), internalNodesEnd = leafValues.end();

            for( ; internalNodesIter != internalNodesEnd; ++internalNodesIter ) // leaves
                leaves.push_back((float)*internalNodesIter);
        }
    }
    return true;
}
