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
#include <openbr_plugin.h>

#include "core/opencvutils.h"

namespace br
{

/*!
 * \ingroup transforms
 * \brief C. Burges. "A tutorial on support vector machines for pattern recognition,"
 * Knowledge Discovery and Data Mining 2(2), 1998.
 * \author Josh Klontz \cite jklontz
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

public:
    /*!
     * \brief The Kernel enum
     */
    enum Kernel { Linear = CvSVM::LINEAR,
                  Poly = CvSVM::POLY,
                  RBF = CvSVM::RBF,
                  Sigmoid = CvSVM::SIGMOID };
    /*!
     * \brief The Type enum
     */
    enum Type { C_SVC = CvSVM::C_SVC,
                NU_SVC = CvSVM::NU_SVC,
                ONE_CLASS = CvSVM::ONE_CLASS,
                EPS_SVR = CvSVM::EPS_SVR,
                NU_SVR = CvSVM::NU_SVR};

private:
    BR_PROPERTY(Kernel, kernel, Linear)
    BR_PROPERTY(Type, type, C_SVC)
    BR_PROPERTY(float, C, -1)
    BR_PROPERTY(float, gamma, -1)

    cv::SVM svm;
    float a, b;

public:
    SVMTransform() : a(1), b(0) {}

private:
    void train(const TemplateList &_data)
    {
        cv::Mat data = OpenCVUtils::toMat(_data.data());
        cv::Mat lab = OpenCVUtils::toMat(_data.labels<float>());

        // Scale labels to [-1,1]
        double min, max;
        cv::minMaxLoc(lab, &min, &max);
        if (max > min) {
            a = 2.0/(max-min);
            b = -(min*a+1);
            lab = (lab * a) + b;
            cv::minMaxLoc(lab, &min, &max);
        }

        if (data.type() != CV_32FC1)
            qFatal("Expected single channel floating point training data.");

        CvSVMParams params;
        params.kernel_type = kernel;
        params.svm_type = type;
        params.p = 0.1;
        params.nu = 0.5;
        if ((C == -1) || ((gamma == -1) && (int(kernel) != int(CvSVM::LINEAR)))) {
            try {
                svm.train_auto(data, lab, cv::Mat(), cv::Mat(), params, 5);
            } catch (...) {
                qWarning("Some classes do not contain sufficient examples or are not discriminative enough for accurate SVM classification.");
                svm.train(data, lab);
            }
        } else {
            params.C = C;
            params.gamma = gamma;
            svm.train(data, lab, cv::Mat(), cv::Mat(), params);
        }

        CvSVMParams p = svm.get_params();
        qDebug("SVM C = %f  Gamma = %f  Support Vectors = %d", p.C, p.gamma, svm.get_support_vector_count());
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        dst.file.setLabel((svm.predict(src.m().reshape(0, 1)) - b)/a);
    }

    void store(QDataStream &stream) const
    {
        stream << a << b;

        // Create local file
        QTemporaryFile tempFile;
        tempFile.open();
        tempFile.close();

        // Save SVM to local file
        svm.save(qPrintable(tempFile.fileName()));

        // Copy local file contents to stream
        tempFile.open();
        QByteArray data = tempFile.readAll();
        tempFile.close();
        stream << data;
    }

    void load(QDataStream &stream)
    {
        stream >> a >> b;

        // Copy local file contents from stream
        QByteArray data;
        stream >> data;

        // Create local file
        QTemporaryFile tempFile(QDir::tempPath()+"/SVM");
        tempFile.open();
        tempFile.write(data);
        tempFile.close();

        // Load SVM from local file
        svm.load(qPrintable(tempFile.fileName()));
    }
};

BR_REGISTER(Transform, SVMTransform)

} // namespace br

#include "svm.moc"
