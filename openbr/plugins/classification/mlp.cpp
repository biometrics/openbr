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

#include <opencv2/ml/ml.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV's multi-layer perceptron framework
 * \author Scott Klum \cite sklum
 * \br_link http://docs.opencv.org/modules/ml/doc/neural_networks.html
 * \br_property enum kernel Type of MLP kernel to use. Options are Identity, Sigmoid, Gaussian. Default is Sigmoid.
 * \br_property float alpha Determines activation function for neural network. See OpenCV documentation for more details. Default is 1.
 * \br_property float beta Determines activation function for neural network. See OpenCV documentation for more details. Default is 1.
 * \br_property QStringList inputVariables Metadata keys for the labels associated with each template. There should be the same number of keys in the list as there are neurons in the final layer. Default is QStringList().
 * \br_property QStringList outputVariables Metadata keys to store the output of the neural network. There should be the same number of keys in the list as there are neurons in the final layer. Default is QStringList().
 * \br_property QList<int> neuronsPerLayer The number of neurons in each layer of the net. Default is QList<int>() << 1 << 1.
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
        OpenCVUtils::loadModel(mlp, stream);
    }

    void store(QDataStream &stream) const
    {
        OpenCVUtils::storeModel(mlp, stream);
    }
};

BR_REGISTER(Transform, MLPTransform)

} // namespace br

#include "classification/mlp.moc"
