#include <opencv2/ml/ml.hpp>

#include "openbr_internal.h"
#include "openbr/core/opencvutils.h"
#include <QString>
#include <QTemporaryFile>

using namespace std;
using namespace cv;

namespace br
{

static void storeModel(const CvStatModel &model, QDataStream &stream)
{
    // Create local file
    QTemporaryFile tempFile;
    tempFile.open();
    tempFile.close();

    // Save MLP to local file
    model.save(qPrintable(tempFile.fileName()));

    // Copy local file contents to stream
    tempFile.open();
    QByteArray data = tempFile.readAll();
    tempFile.close();
    stream << data;
}

static void loadModel(CvStatModel &model, QDataStream &stream)
{
    // Copy local file contents from stream
    QByteArray data;
    stream >> data;

    // Create local file
    QTemporaryFile tempFile(QDir::tempPath()+"/model");
    tempFile.open();
    tempFile.write(data);
    tempFile.close();

    // Load MLP from local file
    model.load(qPrintable(tempFile.fileName()));
}

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV's random trees framework
 * \author Scott Klum \cite sklum
 * \brief http://docs.opencv.org/modules/ml/doc/random_trees.html
 */
class ForestTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(bool classification READ get_classification WRITE set_classification RESET reset_classification STORED false)
    Q_PROPERTY(float splitPercentage READ get_splitPercentage WRITE set_splitPercentage RESET reset_splitPercentage STORED false)
    Q_PROPERTY(int maxDepth READ get_maxDepth WRITE set_maxDepth RESET reset_maxDepth STORED false)
    Q_PROPERTY(int maxTrees READ get_maxTrees WRITE set_maxTrees RESET reset_maxTrees STORED false)
    Q_PROPERTY(float forestAccuracy READ get_forestAccuracy WRITE set_forestAccuracy RESET reset_forestAccuracy STORED false)
    Q_PROPERTY(bool returnConfidence READ get_returnConfidence WRITE set_returnConfidence RESET reset_returnConfidence STORED false)
    Q_PROPERTY(bool overwriteMat READ get_overwriteMat WRITE set_overwriteMat RESET reset_overwriteMat STORED false)
    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED false)
    Q_PROPERTY(QString outputVariable READ get_outputVariable WRITE set_outputVariable RESET reset_outputVariable STORED false)
    BR_PROPERTY(bool, classification, true)
    BR_PROPERTY(float, splitPercentage, .01)
    BR_PROPERTY(int, maxDepth, std::numeric_limits<int>::max())
    BR_PROPERTY(int, maxTrees, 10)
    BR_PROPERTY(float, forestAccuracy, .1)
    BR_PROPERTY(bool, returnConfidence, true)
    BR_PROPERTY(bool, overwriteMat, true)
    BR_PROPERTY(QString, inputVariable, "Label")
    BR_PROPERTY(QString, outputVariable, "")

    CvRTrees forest;
    int totalSize;
    QList< QList<const CvDTreeNode*> > nodes;

    void train(const TemplateList &data)
    {
        Mat samples = OpenCVUtils::toMat(data.data());
        Mat labels = OpenCVUtils::toMat(File::get<float>(data, inputVariable));

        Mat types = Mat(samples.cols + 1, 1, CV_8U);
        types.setTo(Scalar(CV_VAR_NUMERICAL));

        if (classification) {
            types.at<char>(samples.cols, 0) = CV_VAR_CATEGORICAL;
        } else {
            types.at<char>(samples.cols, 0) = CV_VAR_NUMERICAL;
        }

        int minSamplesForSplit = data.size()*splitPercentage;
        forest.train( samples, CV_ROW_SAMPLE, labels, Mat(), Mat(), types, Mat(),
                    CvRTParams(maxDepth,
                               minSamplesForSplit,
                               0,
                               false,
                               2,
                               0,
                               false,
                               0,
                               maxTrees,
                               forestAccuracy,
                               CV_TERMCRIT_ITER));

        qDebug() << "Number of trees:" << forest.get_tree_count();

        for (int i=0; i<forest.get_tree_count(); i++) {
            nodes.append(QList<const CvDTreeNode*>());
            const CvDTreeNode* node = forest.get_tree(i)->get_root();

            // traverse the tree and save all the nodes in depth-first order
            for(;;)
            {
                CvDTreeNode* parent;
                for(;;)
                {
                    if( !node->left )
                        break;
                    node = node->left;
                }

                nodes.last().append(node);

                for( parent = node->parent; parent && parent->right == node;
                    node = parent, parent = parent->parent )
                    ;

                if( !parent )
                    break;

                node = parent->right;
            }

            totalSize += nodes.last().size();
        }
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        /*
        float response;
        if (classification && returnConfidence) {
            // Fuzzy class label
            response = forest.predict_prob(src.m().reshape(1,1));
        } else {
            response = forest.predict(src.m().reshape(1,1));
        }*/

       // QTime timer;
       // timer.start();

        //qDebug() << forest.get_tree(0)->get_var_count();

        Mat responses = Mat::zeros(totalSize,1,CV_32F);

        int offset = 0;
        for (int i=0; i<nodes.size(); i++) {
            int index = nodes[i].indexOf(forest.get_tree(i)->predict(src.m().reshape(1,1)));
            responses.at<float>(offset+index,0) = 1;
            offset += nodes[i].size();
        }

        if (overwriteMat) {
            dst.m() = responses;
            //dst.m() = Mat(1, 1, CV_32F);
            //dst.m().at<float>(0, 0) = response;
        } else {
            //dst.file.set(outputVariable, response);
        }

        //qDebug() << timer.elapsed();
    }

    void load(QDataStream &stream)
    {
        loadModel(forest,stream);
        for (int i=0; i<forest.get_tree_count(); i++) {
            nodes.append(QList<const CvDTreeNode*>());
            const CvDTreeNode* node = forest.get_tree(i)->get_root();

            // traverse the tree and save all the nodes in depth-first order
            for(;;)
            {
                CvDTreeNode* parent;
                for(;;)
                {
                    if( !node->left )
                        break;
                    node = node->left;
                }

                nodes.last().append(node);

                for( parent = node->parent; parent && parent->right == node;
                    node = parent, parent = parent->parent )
                    ;

                if( !parent )
                    break;

                node = parent->right;
            }

            totalSize += nodes.last().size();
        }
    }

    void store(QDataStream &stream) const
    {
        storeModel(forest,stream);
    }

    void init()
    {
        totalSize = 0;

        if (outputVariable.isEmpty())
            outputVariable = inputVariable;
    }
};

BR_REGISTER(Transform, ForestTransform)

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV's Ada Boost framework
 * \author Scott Klum \cite sklum
 * \brief http://docs.opencv.org/modules/ml/doc/boosting.html
 */
class AdaBoostTransform : public Transform
{
    Q_OBJECT
    Q_ENUMS(Type)
    Q_ENUMS(SplitCriteria)

    Q_PROPERTY(Type type READ get_type WRITE set_type RESET reset_type STORED false)
    Q_PROPERTY(SplitCriteria splitCriteria READ get_splitCriteria WRITE set_splitCriteria RESET reset_splitCriteria STORED false)
    Q_PROPERTY(int weakCount READ get_weakCount WRITE set_weakCount RESET reset_weakCount STORED false)
    Q_PROPERTY(float trimRate READ get_trimRate WRITE set_trimRate RESET reset_trimRate STORED false)
    Q_PROPERTY(int folds READ get_folds WRITE set_folds RESET reset_folds STORED false)
    Q_PROPERTY(int maxDepth READ get_maxDepth WRITE set_maxDepth RESET reset_maxDepth STORED false)
    Q_PROPERTY(bool returnConfidence READ get_returnConfidence WRITE set_returnConfidence RESET reset_returnConfidence STORED false)
    Q_PROPERTY(bool overwriteMat READ get_overwriteMat WRITE set_overwriteMat RESET reset_overwriteMat STORED false)
    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED false)
    Q_PROPERTY(QString outputVariable READ get_outputVariable WRITE set_outputVariable RESET reset_outputVariable STORED false)

public:
    enum Type { Discrete = CvBoost::DISCRETE,
                Real = CvBoost::REAL,
                Logit = CvBoost::LOGIT,
                Gentle = CvBoost::GENTLE};

    enum SplitCriteria { Default = CvBoost::DEFAULT,
                         Gini = CvBoost::GINI,
                         Misclass = CvBoost::MISCLASS,
                         Sqerr = CvBoost::SQERR};

private:
    BR_PROPERTY(Type, type, Real)
    BR_PROPERTY(SplitCriteria, splitCriteria, Default)
    BR_PROPERTY(int, weakCount, 100)
    BR_PROPERTY(float, trimRate, .95)
    BR_PROPERTY(int, folds, 0)
    BR_PROPERTY(int, maxDepth, 1)
    BR_PROPERTY(bool, returnConfidence, true)
    BR_PROPERTY(bool, overwriteMat, true)
    BR_PROPERTY(QString, inputVariable, "Label")
    BR_PROPERTY(QString, outputVariable, "")

    CvBoost boost;

    void train(const TemplateList &data)
    {
        Mat samples = OpenCVUtils::toMat(data.data());
        Mat labels = OpenCVUtils::toMat(File::get<float>(data, inputVariable));

        Mat types = Mat(samples.cols + 1, 1, CV_8U);
        types.setTo(Scalar(CV_VAR_NUMERICAL));
        types.at<char>(samples.cols, 0) = CV_VAR_CATEGORICAL;

        CvBoostParams params;
        params.boost_type = type;
        params.split_criteria = splitCriteria;
        params.weak_count = weakCount;
        params.weight_trim_rate = trimRate;
        params.cv_folds = folds;
        params.max_depth = maxDepth;

        boost.train( samples, CV_ROW_SAMPLE, labels, Mat(), Mat(), types, Mat(),
                    params);
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        float response;
        if (returnConfidence) {
            response = boost.predict(src.m().reshape(1,1),Mat(),Range::all(),false,true)/weakCount;
        } else {
            response = boost.predict(src.m().reshape(1,1));
        }

        if (overwriteMat) {
            dst.m() = Mat(1, 1, CV_32F);
            dst.m().at<float>(0, 0) = response;
        } else {
            dst.file.set(outputVariable, response);
        }
    }

    void load(QDataStream &stream)
    {
        loadModel(boost,stream);
    }

    void store(QDataStream &stream) const
    {
        storeModel(boost,stream);
    }


    void init()
    {
        if (outputVariable.isEmpty())
            outputVariable = inputVariable;
    }
};

BR_REGISTER(Transform, AdaBoostTransform)

} // namespace br

#include "tree.moc"
