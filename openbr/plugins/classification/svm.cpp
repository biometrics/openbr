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

#include <QTemporaryFile>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV's SVM framework.
 * \br_link http://docs.opencv.org/modules/ml/doc/support_vector_machines.html
 * \br_paper C. Burges.
 *           "A tutorial on support vector machines for pattern recognition"
 *           Knowledge Discovery and Data Mining 2(2), 1998.
 * \author Josh Klontz \cite jklontz
 *
 * \br_property enum Kernel The type of SVM kernel to use. Options are Linear, Poly, RBF, Sigmoid. Default is Linear.
 * \br_property enum Type The type of SVM to do. Options are C_SVC, NU_SVC, ONE_CLASS, EPS_SVR, NU_SVR. Default is C_SVC.
 * \br_property float C Parameter C of an SVM optimization problem. Needed when Type is C_SVC, EPS_SVR or NU_SVR. Default is -1.
 * \br_property float gamma Parameter gamma of a kernel function. Needed when Kernel is Poly, RBF, or Sigmoid. Default is -1.
 * \br_property QString inputVariable Metadata variable storing the label for each template. Default is "Label".
 * \br_property QString outputVariable Metadata variable to store the prediction value of the trained SVM. If type is EPS_SVR or NU_SVR the stored value is the output of the SVM. Otherwise the value is the output of the SVM mapped through the reverse lookup table. Default is "".
 * \br_property bool returnDFVal If true, dst is set to a 1x1 Mat with value equal to the predicted output of the SVM. Default is false.
 * \br_property int termCriteria The maximum number of training iterations. Default is 1000.
 * \br_property int folds Cross validation parameter used for autoselecting other parameters. Default is 5.
 * \br_property bool balanceFolds If true and the problem is 2-class classification then more balanced cross validation subsets are created. Default is false.
 */
class SVMTransform : public Transform
{
    Q_OBJECT
    Q_ENUMS(Kernel)
    Q_ENUMS(Type)
    Q_PROPERTY(Kernel kernel READ get_kernel WRITE set_kernel RESET reset_kernel STORED false)
    Q_PROPERTY(Type type READ get_type WRITE set_type RESET reset_type STORED false)
    Q_PROPERTY(float C READ get_C WRITE set_C RESET reset_C STORED false)
    Q_PROPERTY(float gamma READ get_gamma WRITE set_gamma RESET reset_gamma STORED false)
    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED false)
    Q_PROPERTY(QString outputVariable READ get_outputVariable WRITE set_outputVariable RESET reset_outputVariable STORED false)
    Q_PROPERTY(bool returnDFVal READ get_returnDFVal WRITE set_returnDFVal RESET reset_returnDFVal STORED false)
    Q_PROPERTY(int termCriteria READ get_termCriteria WRITE set_termCriteria RESET reset_termCriteria STORED false)
    Q_PROPERTY(int folds READ get_folds WRITE set_folds RESET reset_folds STORED false)
    Q_PROPERTY(bool balanceFolds READ get_balanceFolds WRITE set_balanceFolds RESET reset_balanceFolds STORED false)

public:
    enum Kernel { Linear = CvSVM::LINEAR,
                  Poly = CvSVM::POLY,
                  RBF = CvSVM::RBF,
                  Sigmoid = CvSVM::SIGMOID };

    enum Type { C_SVC = CvSVM::C_SVC,
                NU_SVC = CvSVM::NU_SVC,
                ONE_CLASS = CvSVM::ONE_CLASS,
                EPS_SVR = CvSVM::EPS_SVR,
                NU_SVR = CvSVM::NU_SVR};

    SVM svm;

private:
    BR_PROPERTY(Kernel, kernel, Linear)
    BR_PROPERTY(Type, type, C_SVC)
    BR_PROPERTY(float, C, -1)
    BR_PROPERTY(float, gamma, -1)
    BR_PROPERTY(QString, inputVariable, "Label")
    BR_PROPERTY(QString, outputVariable, "")
    BR_PROPERTY(bool, returnDFVal, false)
    BR_PROPERTY(int, termCriteria, 1000)
    BR_PROPERTY(int, folds, 5)
    BR_PROPERTY(bool, balanceFolds, false)

    QHash<QString, int> labelMap;
    QHash<int, QVariant> reverseLookup;

    void train(const TemplateList &_data)
    {
        Mat data = OpenCVUtils::toMat(_data.data());
        Mat lab;
        // If we are doing regression, the input variable should have float
        // values
        if (type == EPS_SVR || type == NU_SVR) {
            lab = OpenCVUtils::toMat(File::get<float>(_data, inputVariable));
        }
        // If we are doing classification, we should be dealing with discrete
        // values. Map them and store the mapping data
        else {
            QList<int> dataLabels = _data.indexProperty(inputVariable, labelMap, reverseLookup);
            lab = OpenCVUtils::toMat(dataLabels);
        }

        if (data.type() != CV_32FC1)
            qFatal("Expected single channel floating point training data.");

        CvSVMParams params;
        params.kernel_type = kernel;
        params.svm_type = type;
        params.p = 0.1;
        params.nu = 0.5;
        params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, termCriteria, FLT_EPSILON);

        if ((C == -1) || ((gamma == -1) && (kernel == RBF))) {
            try {
                svm.train_auto(data, lab, Mat(), Mat(), params, folds,
                               CvSVM::get_default_grid(CvSVM::C),
                               CvSVM::get_default_grid(CvSVM::GAMMA),
                               CvSVM::get_default_grid(CvSVM::P),
                               CvSVM::get_default_grid(CvSVM::NU),
                               CvSVM::get_default_grid(CvSVM::COEF),
                               CvSVM::get_default_grid(CvSVM::DEGREE),
                               balanceFolds);
            } catch (...) {
                qWarning("Some classes do not contain sufficient examples or are not discriminative enough for accurate SVM classification.");
                svm.train(data, lab, Mat(), Mat(), params);
            }
        } else {
            params.C = C;
            params.gamma = gamma;
            svm.train(data, lab, Mat(), Mat(), params);
        }

        CvSVMParams p = svm.get_params();
        qDebug("SVM C = %f  Gamma = %f  Support Vectors = %d", p.C, p.gamma, svm.get_support_vector_count());
    }

    void project(const Template &src, Template &dst) const
    {
        if (returnDFVal && reverseLookup.size() > 2)
            qFatal("Decision function for multiclass classification not implemented.");

        dst = src;
        float prediction = svm.predict(src.m().reshape(1, 1), returnDFVal);
        if (returnDFVal) {
            dst.m() = Mat(1, 1, CV_32F);
            dst.m().at<float>(0, 0) = prediction;
            // positive values ==> first class
            // negative values ==> second class
            if (type != EPS_SVR && type != NU_SVR)
                prediction = prediction > 0 ? 0 : 1;
        }
        if (type == EPS_SVR || type == NU_SVR) {
            dst.file.set(outputVariable, prediction);
            dst.m() = Mat(1, 1, CV_32F);
            dst.m().at<float>(0, 0) = prediction;

        } else
            dst.file.set(outputVariable, reverseLookup[prediction]);
    }

    void store(QDataStream &stream) const
    {
        OpenCVUtils::storeModel(svm, stream);
        stream << labelMap << reverseLookup;
    }

    void load(QDataStream &stream)
    {
        OpenCVUtils::loadModel(svm, stream);
        stream >> labelMap >> reverseLookup;
    }

    void init()
    {
        if (outputVariable.isEmpty())
            outputVariable = inputVariable;
    }
};

BR_REGISTER(Transform, SVMTransform)

// Hack to expose the underlying SVM as it is difficult to expose this data structure through the Qt property system
BR_EXPORT const cv::SVM *GetSVM(const br::Transform *t)
{
    return &reinterpret_cast<const SVMTransform&>(*t).svm;
}

} // namespace br

#include "classification/svm.moc"
