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
 * \brief Wraps OpenCV's Ada Boost framework
 * \author Scott Klum \cite sklum
 * \br_link http://docs.opencv.org/modules/ml/doc/boosting.html
 * \br_property type enum Type of Adaboost to perform. Options are: [Discrete, Real, Logit, Gentle] Default is Real.
 * \br_property splitCriteria enum Splitting criteria used to choose optimal splits during a weak tree construction. Options are: [Default, Gini, Misclass, Sqerr] Default is Default.
 * \br_property weakCount int Maximum number of weak classifiers per stage. Default is 100.
 * \br_property trimRate float A threshold between 0 and 1 used to save computational time. Samples with summary weight \leq 1 - weight\_trim\_rate do not participate in the next iteration of training. Set this parameter to 0 to turn off this functionality. Default is 0.95.
 * \br_property folds int OpenCV parameter variable. Default value is 0.
 * \br_property maxDepth int Maximum height of each weak classifier tree. Default is 1 (stumps).
 * \br_property returnConfidence bool Return the confidence value of the classification or the class value of the classification. Default is true (return confidence value).
 * \br_property overwriteMat bool If true, the output template will be a 1x1 matrix with value equal to the confidence or classification (depending on returnConfidence). If false the output template will be the same as the input template. Default is true.
 * \br_property inputVariable QString Metadata variable storing the label for each template. Default is "Label".
 * \br_property outputVariable QString Metadata variable to store the confidence or classification of each template (depending on returnConfidence). If overwriteMat is true nothing will be written here. Default is "".
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
        OpenCVUtils::loadModel(boost,stream);
    }

    void store(QDataStream &stream) const
    {
        OpenCVUtils::storeModel(boost,stream);
    }


    void init()
    {
        if (outputVariable.isEmpty())
            outputVariable = inputVariable;
    }
};

BR_REGISTER(Transform, AdaBoostTransform)

} // namespace br

#include "classification/adaboost.moc"
