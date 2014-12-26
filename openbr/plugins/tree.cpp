#include <opencv2/ml/ml.hpp>

#include "openbr_internal.h"
#include "openbr/core/opencvutils.h"
#include <QString>
#include <QTemporaryFile>

using namespace std;
using namespace cv;

namespace br
{

static void storeForest(const CvRTrees &forest, QDataStream &stream)
{
    // Create local file
    QTemporaryFile tempFile;
    tempFile.open();
    tempFile.close();

    // Save MLP to local file
    forest.save(qPrintable(tempFile.fileName()));

    // Copy local file contents to stream
    tempFile.open();
    QByteArray data = tempFile.readAll();
    tempFile.close();
    stream << data;
}

static void loadForest(CvRTrees &forest, QDataStream &stream)
{
    // Copy local file contents from stream
    QByteArray data;
    stream >> data;

    // Create local file
    QTemporaryFile tempFile(QDir::tempPath()+"/forest");
    tempFile.open();
    tempFile.write(data);
    tempFile.close();

    // Load MLP from local file
    forest.load(qPrintable(tempFile.fileName()));
}

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV's random trees framework
 * \author Scott Klum \cite sklum
 * \brief http://docs.opencv.org/modules/ml/doc/random_trees.html
 */
class ForestTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(bool classification READ get_classification WRITE set_classification RESET reset_classification STORED true)
    Q_PROPERTY(float splitPercentage READ get_splitPercentage WRITE set_splitPercentage RESET reset_splitPercentage STORED true)
    Q_PROPERTY(int maxDepth READ get_maxDepth WRITE set_maxDepth RESET reset_maxDepth STORED true)
    Q_PROPERTY(int maxTrees READ get_maxTrees WRITE set_maxTrees RESET reset_maxTrees STORED true)
    Q_PROPERTY(float forestAccuracy READ get_forestAccuracy WRITE set_forestAccuracy RESET reset_forestAccuracy STORED true)
    BR_PROPERTY(bool, classification, true)
    BR_PROPERTY(float, splitPercentage, .01)
    BR_PROPERTY(int, maxDepth, std::numeric_limits<int>::max())
    BR_PROPERTY(int, maxTrees, 10)
    BR_PROPERTY(float, forestAccuracy, .1)

    CvRTrees forest;

    void train(const TemplateList &data)
    {
        Mat samples = OpenCVUtils::toMat(data.data());
        Mat labels = OpenCVUtils::toMat(File::get<float>(data, "Label"));

        Mat types = Mat(samples.cols + 1, 1, CV_8U );
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
                               0, // priors
                               false,
                               0,
                               maxTrees,
                               forestAccuracy,
                               CV_TERMCRIT_EPS));

        qDebug() << "Number of trees:" << forest.get_tree_count();
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        float response = forest.predict_prob(src.m().reshape(1,1));
        dst.m() = Mat(1, 1, CV_32F);
        dst.m().at<float>(0, 0) = response;
    }

    void load(QDataStream &stream)
    {
        loadForest(forest,stream);
    }

    void store(QDataStream &stream) const
    {
        storeForest(forest,stream);
    }
};

BR_REGISTER(Transform, ForestTransform)

} // namespace br

#include "tree.moc"
