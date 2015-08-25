#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup Distances
 * \brief SVM Regression on Template absolute differences.
 * \author Josh Klontz \cite jklontz
 */
class SVMDistance : public Distance
{
    Q_OBJECT
    Q_ENUMS(Kernel)
    Q_ENUMS(Type)
    Q_PROPERTY(Kernel kernel READ get_kernel WRITE set_kernel RESET reset_kernel STORED false)
    Q_PROPERTY(Type type READ get_type WRITE set_type RESET reset_type STORED false)
    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED false)
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

private:
    BR_PROPERTY(Kernel, kernel, Linear)
    BR_PROPERTY(Type, type, EPS_SVR)
    BR_PROPERTY(QString, inputVariable, "Label")
    BR_PROPERTY(int, termCriteria, 1000)
    BR_PROPERTY(int, folds, 5)
    BR_PROPERTY(bool, balanceFolds, false)

    SVM svm;

    void train(const TemplateList &src)
    {
        const Mat data = OpenCVUtils::toMat(src.data());
        const QList<int> lab = src.indexProperty(inputVariable);

        const int instances = data.rows * (data.rows+1) / 2;
        Mat deltaData(instances, data.cols, data.type());
        Mat deltaLab(instances, 1, CV_32FC1);
        int index = 0;
        for (int i=0; i<data.rows; i++)
            for (int j=i; j<data.rows; j++) {
                const bool match = lab[i] == lab[j];
                if (!match && (type == ONE_CLASS))
                    continue;
                absdiff(data.row(i), data.row(j), deltaData.row(index));
                deltaLab.at<float>(index, 0) = (match ? 1 : 0);
                index++;
            }
        deltaData = deltaData.rowRange(0, index);
        deltaLab = deltaLab.rowRange(0, index);

        const float C = -1;
        const float gamma = -1;

        if (deltaData.type() != CV_32FC1)
            qFatal("Expected single channel floating point training data.");

        CvSVMParams params;
        params.kernel_type = kernel;
        params.svm_type = type;
        params.p = 0.1;
        params.nu = 0.5;
        params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, termCriteria, FLT_EPSILON);

        if ((C == -1) || ((gamma == -1) && (kernel == RBF))) {
            try {
                svm.train_auto(deltaData, deltaLab, Mat(), Mat(), params, folds,
                               CvSVM::get_default_grid(CvSVM::C),
                               CvSVM::get_default_grid(CvSVM::GAMMA),
                               CvSVM::get_default_grid(CvSVM::P),
                               CvSVM::get_default_grid(CvSVM::NU),
                               CvSVM::get_default_grid(CvSVM::COEF),
                               CvSVM::get_default_grid(CvSVM::DEGREE),
                               balanceFolds);
            } catch (...) {
                qWarning("Some classes do not contain sufficient examples or are not discriminative enough for accurate SVM classification.");
                svm.train(deltaData, deltaLab, Mat(), Mat(), params);
            }
        } else {
            params.C = C;
            params.gamma = gamma;
            svm.train(deltaData, deltaLab, Mat(), Mat(), params);
        }

        CvSVMParams p = svm.get_params();
        qDebug("SVM C = %f  Gamma = %f  Support Vectors = %d", p.C, p.gamma, svm.get_support_vector_count());
    }

    float compare(const Mat &a, const Mat &b) const
    {
        Mat delta;
        absdiff(a, b, delta);
        return svm.predict(delta.reshape(1, 1));
    }

    void store(QDataStream &stream) const
    {
        OpenCVUtils::storeModel(svm, stream);
    }

    void load(QDataStream &stream)
    {
        OpenCVUtils::loadModel(svm, stream);
    }
};

BR_REGISTER(Distance, SVMDistance)

} // namespace br

#include "distance/svm.moc"
