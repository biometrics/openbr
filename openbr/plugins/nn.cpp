#include <opencv2/ml/ml.hpp>

#include "openbr_internal.h"
#include "openbr/core/qtutils.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/eigenutils.h"
#include <QString>
#include <QTemporaryFile>

using namespace std;
using namespace cv;

namespace br
{

static void storeMLP(const CvANN_MLP &mlp, QDataStream &stream)
{
    // Create local file
    QTemporaryFile tempFile;
    tempFile.open();
    tempFile.close();

    // Save MLP to local file
    mlp.save(qPrintable(tempFile.fileName()));

    // Copy local file contents to stream
    tempFile.open();
    QByteArray data = tempFile.readAll();
    tempFile.close();
    stream << data;
}

static void loadMLP(CvANN_MLP &mlp, QDataStream &stream)
{
    // Copy local file contents from stream
    QByteArray data;
    stream >> data;

    // Create local file
    QTemporaryFile tempFile(QDir::tempPath()+"/MLP");
    tempFile.open();
    tempFile.write(data);
    tempFile.close();

    // Load MLP from local file
    mlp.load(qPrintable(tempFile.fileName()));
}

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV's multi-layer perceptron framework
 * \author Scott Klum \cite sklum
 * \brief http://docs.opencv.org/modules/ml/doc/neural_networks.html
 */
class MLPTransform : public MetaTransform
{
    Q_OBJECT

    Q_ENUMS(Kernel)
    Q_PROPERTY(Kernel kernel READ get_kernel WRITE set_kernel RESET reset_kernel STORED false)
    Q_PROPERTY(float alpha READ get_alpha WRITE set_alpha RESET reset_alpha STORED false)
    Q_PROPERTY(float beta READ get_beta WRITE set_beta RESET reset_beta STORED false)
    Q_PROPERTY(QStringList inputVariables READ get_inputVariables WRITE set_inputVariables RESET reset_inputVariables STORED false)
    Q_PROPERTY(QStringList outputVariables READ get_outputVariables WRITE set_outputVariables RESET reset_outputVariables STORED false)
    Q_PROPERTY(QList<int> neuronsPerLayer READ get_neuronsPerLayer WRITE set_neuronsPerLayer RESET reset_neuronsPerLayer STORED false)

public:

    enum Kernel { Identity = CvANN_MLP::IDENTITY,
                  Sigmoid = CvANN_MLP::SIGMOID_SYM,
                  Gaussian = CvANN_MLP::GAUSSIAN};

private:
    BR_PROPERTY(Kernel, kernel, Sigmoid)
    BR_PROPERTY(float, alpha, 1)
    BR_PROPERTY(float, beta, 1)
    BR_PROPERTY(QStringList, inputVariables, QStringList())
    BR_PROPERTY(QStringList, outputVariables, QStringList())
    BR_PROPERTY(QList<int>, neuronsPerLayer, QList<int>() << 1 << 1)

    CvANN_MLP mlp;

    void init()
    {
        if (kernel == Gaussian)
            qWarning("The OpenCV documentation warns that the Gaussian kernel, \"is not completely supported at the moment\"");

        Mat layers = Mat(neuronsPerLayer.size(), 1, CV_32SC1);
        for (int i=0; i<neuronsPerLayer.size(); i++)
            layers.row(i) = Scalar(neuronsPerLayer.at(i));

        mlp.create(layers,kernel, alpha, beta);
    }

    void train(const TemplateList &data)
    {
        Mat _data = OpenCVUtils::toMat(data.data());

        // Assuming data has n templates
        // _data needs to be n x size of input layer
        // Labels needs to be a n x outputs matrix
        // For the time being we're going to assume a single output
        Mat labels = Mat::zeros(data.size(),inputVariables.size(),CV_32F);
        for (int i=0; i<inputVariables.size(); i++)
            labels.col(i) += OpenCVUtils::toMat(File::get<float>(data, inputVariables.at(i)));

        mlp.train(_data,labels,Mat());

        if (Globals->verbose)
            for (int i=0; i<neuronsPerLayer.size(); i++) qDebug() << *mlp.get_weights(i);
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        // See above for response dimensionality
        Mat response(outputVariables.size(), 1, CV_32FC1);
        mlp.predict(src.m().reshape(1,1),response);

        // Apparently mlp.predict reshapes the response matrix?
        for (int i=0; i<outputVariables.size(); i++) dst.file.set(outputVariables.at(i),response.at<float>(0,i));
    }

    void load(QDataStream &stream)
    {
        loadMLP(mlp,stream);
    }

    void store(QDataStream &stream) const
    {
        storeMLP(mlp,stream);
    }
};

BR_REGISTER(Transform, MLPTransform)

} // namespace br

#include "nn.moc"
