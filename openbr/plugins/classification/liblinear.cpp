#include <QTemporaryFile>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

#include <linear.h>

using namespace cv;

namespace br
{

static void storeModel(const model &m, QDataStream &stream)
{
    // Create local file
    QTemporaryFile tempFile;
    tempFile.open();
    tempFile.close();

    // Save MLP to local file
    save_model(qPrintable(tempFile.fileName()),&m);

    // Copy local file contents to stream
    tempFile.open();
    QByteArray data = tempFile.readAll();
    tempFile.close();
    stream << data;
}

static void loadModel(model &m, QDataStream &stream)
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
    m = *load_model(qPrintable(tempFile.fileName()));
}

/*!
 * \brief Wraps LibLinear's Linear SVM framework.
 * \author Scott Klum \cite sklum
 */
class Linear : public Transform
{
    Q_OBJECT
    Q_ENUMS(Solver)
    Q_PROPERTY(Solver solver READ get_solver WRITE set_solver RESET reset_solver STORED false)
    Q_PROPERTY(float C READ get_C WRITE set_C RESET reset_C STORED false)
    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED false)
    Q_PROPERTY(QString outputVariable READ get_outputVariable WRITE set_outputVariable RESET reset_outputVariable STORED false)
    Q_PROPERTY(bool returnDFVal READ get_returnDFVal WRITE set_returnDFVal RESET reset_returnDFVal STORED false)
    Q_PROPERTY(bool overwriteMat READ get_overwriteMat WRITE set_overwriteMat RESET reset_overwriteMat STORED false)
    Q_PROPERTY(bool weight READ get_weight WRITE set_weight RESET reset_weight STORED false)

public:
    enum Solver { L2R_LR = ::L2R_LR,
                  L2R_L2LOSS_SVC_DUAL = ::L2R_L2LOSS_SVC_DUAL,
                  L2R_L2LOSS_SVC = ::L2R_L2LOSS_SVC,
                  L2R_L1LOSS_SVC_DUAL = ::L2R_L1LOSS_SVC_DUAL,
                  MCSVM_CS = ::MCSVM_CS,
                  L1R_L2LOSS_SVC = ::L1R_L2LOSS_SVC,
                  L1R_LR = ::L1R_LR,
                  L2R_LR_DUAL = ::L2R_LR_DUAL,
                  L2R_L2LOSS_SVR = ::L2R_L2LOSS_SVR,
                  L2R_L2LOSS_SVR_DUAL = ::L2R_L2LOSS_SVR_DUAL,
                  L2R_L1LOSS_SVR_DUAL = ::L2R_L1LOSS_SVR_DUAL };

private:
    BR_PROPERTY(Solver, solver, L2R_L2LOSS_SVC_DUAL)
    BR_PROPERTY(float, C, 1)
    BR_PROPERTY(QString, inputVariable, "Label")
    BR_PROPERTY(QString, outputVariable, "")
    BR_PROPERTY(bool, returnDFVal, false)
    BR_PROPERTY(bool, overwriteMat, true)
    BR_PROPERTY(bool, weight, false)

    model m;

    void train(const TemplateList &data)
    {
        Mat samples = OpenCVUtils::toMat(data.data());
        Mat labels = OpenCVUtils::toMat(File::get<float>(data, inputVariable));

        problem prob;
        prob.n = samples.cols;
        prob.l = samples.rows;
        prob.bias = -1;
        prob.y = new double[prob.l];

        for (int i=0; i<prob.l; i++)
            prob.y[i] = labels.at<float>(i,0);

        // Allocate enough memory for l feature_nodes pointers
        prob.x = new feature_node*[prob.l];
        feature_node *x_space = new feature_node[(prob.n+1)*prob.l];

        int k = 0;
        for (int i=0; i<prob.l; i++) {
            prob.x[i] = &x_space[k];
            for (int j=0; j<prob.n; j++) {
                x_space[k].index = j+1;
                x_space[k].value = samples.at<float>(i,j);
                k++;
            }
            x_space[k++].index = -1;
        }

        parameter param;

        // TODO: Support grid search
        param.C = C;
        param.p = 1;
        param.eps = FLT_EPSILON;
        param.solver_type = solver;

        if (weight) {
            param.nr_weight = 2;
            param.weight_label = new int[2];
            param.weight = new double[2];
            param.weight_label[0] = 0;
            param.weight_label[1] = 1;
            int nonZero = countNonZero(labels);
            param.weight[0] = 1;
            param.weight[1] = (double)(prob.l-nonZero)/nonZero;
            qDebug() << param.weight[0] << param.weight[1];
        } else {
            param.nr_weight = 0;
            param.weight_label = NULL;
            param.weight = NULL;
        }

        m = *train_svm(&prob, &param);

        delete[] param.weight;
        delete[] param.weight_label;
        delete[] prob.y;
        delete[] prob.x;
        delete[] x_space;
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        Mat sample = src.m().reshape(1,1);
        feature_node *x_space = new feature_node[sample.cols+1];

        for (int j=0; j<sample.cols; j++) {
            x_space[j].index = j+1;
            x_space[j].value = sample.at<float>(0,j);
        }
        x_space[sample.cols].index = -1;

        float prediction;
        double prob_estimates[m.nr_class];

        if (solver == L2R_L2LOSS_SVR        ||
            solver == L2R_L1LOSS_SVR_DUAL   ||
            solver == L2R_L2LOSS_SVR_DUAL   ||
            solver == L2R_L2LOSS_SVC_DUAL   ||
            solver == L2R_L2LOSS_SVC        ||
            solver == L2R_L1LOSS_SVC_DUAL   ||
            solver == MCSVM_CS              ||
            solver == L1R_L2LOSS_SVC)
        {
            prediction = predict_values(&m,x_space,prob_estimates);
            if (returnDFVal) prediction = prob_estimates[0];
        } else if (solver == L2R_LR         ||
                   solver == L2R_LR_DUAL    ||
                   solver == L1R_LR)
        {
            prediction = predict_probability(&m,x_space,prob_estimates);
            if (returnDFVal) prediction = prob_estimates[0];
        }

        if (overwriteMat) {
            dst.m() = Mat(1, 1, CV_32F);
            dst.m().at<float>(0, 0) = prediction;
        } else {
            dst.file.set(outputVariable,prediction);
        }

        delete[] x_space;
    }

    void store(QDataStream &stream) const
    {
        storeModel(m,stream);
    }

    void load(QDataStream &stream)
    {
        loadModel(m,stream);
    }
};

BR_REGISTER(Transform, Linear)

} // namespace br

#include "liblinear.moc"
