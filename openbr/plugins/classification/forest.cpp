/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2012 The MITRE Corporation                                      *
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License");           *
 * you may not use this file except in compliance with the License.          *
 * You may obtain a copy of the License at                                   *
 *                                                                           *
 *     http://www.apache.org/licenses/LICENSE-2.0                            *
 *                                                                           *
 * Unless required by applicable law or agreed to in writing, software       *
 * distributed under the License is distributed on an "AS IS" BASIS,         *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 * See the License for the specific language governing permissions and       *
 * limitations under the License.                                            *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV's random trees framework
 * \author Scott Klum \cite sklum
 * \br_link http://docs.opencv.org/modules/ml/doc/random_trees.html
 * \br_property bool classification If true the labels are expected to be categorical. Otherwise they are expected to be numerical. Default is true.
 * \br_property float splitPercentage Used to calculate the minimum number of samples per split in a random tree. The minimum number of samples is calculated as the number of samples x splitPercentage. Default is 0.01.
 * \br_property int maxDepth The maximum depth of each decision tree. Default is std::numeric_limits<int>::max() and typically should be set by the user.
 * \br_property int maxTrees The maximum number of trees in the forest. Default is 10.
 * \br_property float forestAccuracy A sufficient accuracy for the forest for training to terminate. Used if termCrit is EPS or Both. Default is 0.1.
 * \br_property bool returnConfidence If both classification and returnConfidence are use a fuzzy class label as the output of the forest. Default is true.
 * \br_property bool overwriteMat If true set dst to be a 1x1 Mat with the forest response as its value. Otherwise append the forest response to metadata using outputVariable as a key. Default is true.
 * \br_property QString inputVariable The metadata key for each templates label. Default is "Label".
 * \br_property QString outputVariable The metadata key for the forest response if overwriteMat is false. Default is "".
 * \br_property bool weight If true and classification is true the random forest will use prior accuracies. Default is false.
 * \br_property enum termCrit Termination criteria for training the random forest. Options are Iter, EPS and Both. Iter terminates when the maximum number of trees is reached. EPS terminates when forestAccuracy is met. Both terminates when either is true. Default is Iter.
 */
class ForestTransform : public Transform
{
    Q_OBJECT

    void train(const TemplateList &data)
    {
        trainForest(data);
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        float response;
        if (classification && returnConfidence) {
            // Fuzzy class label
            response = forest.predict_prob(src.m().reshape(1,1));
        } else {
            response = forest.predict(src.m().reshape(1,1));
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
        OpenCVUtils::loadModel(forest,stream);
    }

    void store(QDataStream &stream) const
    {
        OpenCVUtils::storeModel(forest,stream);
    }

    void init()
    {
        if (outputVariable.isEmpty())
            outputVariable = inputVariable;
    }

protected:
    Q_ENUMS(TerminationCriteria)
    Q_PROPERTY(bool classification READ get_classification WRITE set_classification RESET reset_classification STORED false)
    Q_PROPERTY(float splitPercentage READ get_splitPercentage WRITE set_splitPercentage RESET reset_splitPercentage STORED false)
    Q_PROPERTY(int maxDepth READ get_maxDepth WRITE set_maxDepth RESET reset_maxDepth STORED false)
    Q_PROPERTY(int maxTrees READ get_maxTrees WRITE set_maxTrees RESET reset_maxTrees STORED false)
    Q_PROPERTY(float forestAccuracy READ get_forestAccuracy WRITE set_forestAccuracy RESET reset_forestAccuracy STORED false)
    Q_PROPERTY(bool returnConfidence READ get_returnConfidence WRITE set_returnConfidence RESET reset_returnConfidence STORED false)
    Q_PROPERTY(bool overwriteMat READ get_overwriteMat WRITE set_overwriteMat RESET reset_overwriteMat STORED false)
    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED false)
    Q_PROPERTY(QString outputVariable READ get_outputVariable WRITE set_outputVariable RESET reset_outputVariable STORED false)
    Q_PROPERTY(bool weight READ get_weight WRITE set_weight RESET reset_weight STORED false)
    Q_PROPERTY(TerminationCriteria termCrit READ get_termCrit WRITE set_termCrit RESET reset_termCrit STORED false)

public:
    enum TerminationCriteria { Iter = CV_TERMCRIT_ITER,
                EPS = CV_TERMCRIT_EPS,
                Both = CV_TERMCRIT_EPS | CV_TERMCRIT_ITER};

protected:
    BR_PROPERTY(bool, classification, true)
    BR_PROPERTY(float, splitPercentage, .01)
    BR_PROPERTY(int, maxDepth, std::numeric_limits<int>::max())
    BR_PROPERTY(int, maxTrees, 10)
    BR_PROPERTY(float, forestAccuracy, .1)
    BR_PROPERTY(bool, returnConfidence, true)
    BR_PROPERTY(bool, overwriteMat, true)
    BR_PROPERTY(QString, inputVariable, "Label")
    BR_PROPERTY(QString, outputVariable, "")
    BR_PROPERTY(bool, weight, false)
    BR_PROPERTY(TerminationCriteria, termCrit, Iter)

    CvRTrees forest;

    void trainForest(const TemplateList &data)
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

        bool usePrior = classification && weight;
        float priors[2];
        if (usePrior) {
            int nonZero = countNonZero(labels);
            priors[0] = 1;
            priors[1] = (float)(samples.rows-nonZero)/nonZero;
        }

        int minSamplesForSplit = data.size()*splitPercentage;
        forest.train( samples, CV_ROW_SAMPLE, labels, Mat(), Mat(), types, Mat(),
                    CvRTParams(maxDepth,
                               minSamplesForSplit,
                               0,
                               false,
                               2,
                               usePrior ? priors : 0,
                               false,
                               0,
                               maxTrees,
                               forestAccuracy,
                               termCrit));

        if (Globals->verbose) {
            qDebug() << "Number of trees:" << forest.get_tree_count();

            if (classification) {
                QTime timer;
                timer.start();
                int correctClassification = 0;
                float regressionError = 0;
                for (int i=0; i<samples.rows; i++) {
                    float prediction = forest.predict_prob(samples.row(i));
                    int label = forest.predict(samples.row(i));
                    if (label == labels.at<float>(i,0)) {
                        correctClassification++;
                    }
                    regressionError += fabs(prediction-labels.at<float>(i,0));
                }

                qDebug("Time to classify %d samples: %d ms\n \
                       Classification Accuracy: %f\n \
                       MAE: %f\n \
                       Sample dimensionality: %d",
                       samples.rows,timer.elapsed(),(float)correctClassification/samples.rows,regressionError/samples.rows,samples.cols);
            }
        }
    }
};

BR_REGISTER(Transform, ForestTransform)

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV's random trees framework to induce features
 * \author Scott Klum \cite sklum
 * \br_link https://lirias.kuleuven.be/bitstream/123456789/316661/1/icdm11-camready.pdf
 * \br_property bool useRegressionValue SCOTT FILL ME IN.
 */
class ForestInductionTransform : public ForestTransform
{
    Q_OBJECT
    Q_PROPERTY(bool useRegressionValue READ get_useRegressionValue WRITE set_useRegressionValue RESET reset_useRegressionValue STORED false)
    BR_PROPERTY(bool, useRegressionValue, false)

    int totalSize;
    QList< QList<const CvDTreeNode*> > nodes;

    void fillNodes()
    {
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

    void train(const TemplateList &data)
    {
        trainForest(data);
        if (!useRegressionValue) fillNodes();
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        Mat responses;

        if (useRegressionValue) {
            responses = Mat::zeros(forest.get_tree_count(),1,CV_32F);
            for (int i=0; i<forest.get_tree_count(); i++) {
                responses.at<float>(i,0) = forest.get_tree(i)->predict(src.m().reshape(1,1))->value;
            }
        } else {
            responses = Mat::zeros(totalSize,1,CV_32F);
            int offset = 0;
            for (int i=0; i<nodes.size(); i++) {
                int index = nodes[i].indexOf(forest.get_tree(i)->predict(src.m().reshape(1,1)));
                responses.at<float>(offset+index,0) = 1;
                offset += nodes[i].size();
            }
        }

        dst.m() = responses;
    }

    void load(QDataStream &stream)
    {
        OpenCVUtils::loadModel(forest,stream);
        if (!useRegressionValue) fillNodes();

    }

    void store(QDataStream &stream) const
    {
        OpenCVUtils::storeModel(forest,stream);
    }
};

BR_REGISTER(Transform, ForestInductionTransform)

} // namespace br

#include "classification/forest.moc"
