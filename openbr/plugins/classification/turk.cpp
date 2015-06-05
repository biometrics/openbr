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

namespace br
{

/*!
 * \ingroup transforms
 * \brief Convenience class for training turk attribute regressors
 * \author Josh Klontz \cite jklontz
 * \br_property QString key Metadata key to pass input values to SVM. Actual lookup key is "key_value" where value is each value in the parameter values. Default is "".
 * \br_property QStringList values Metadata keys to pass input values to SVM. Actual lookup key is "key_value" where key is the parameter key and value is each value in this list. Each passed value trains a new SVM with the input values found in metadata["key_value"]. Default is "".
 * \br_property bool isMeta If true, "Average+SaveMat(predicted_key_value)" is appended to each classifier. If false, nothing is appended. Default is false.
 */
class TurkClassifierTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(QString key READ get_key WRITE set_key RESET reset_key STORED false)
    Q_PROPERTY(QStringList values READ get_values WRITE set_values RESET reset_values STORED false)
    Q_PROPERTY(bool isMeta READ get_isMeta WRITE set_isMeta RESET reset_isMeta STORED false)
    BR_PROPERTY(QString, key, QString())
    BR_PROPERTY(QStringList, values, QStringList())
    BR_PROPERTY(bool, isMeta, false)

    Transform *child;

    void init()
    {
        QStringList classifiers;
        foreach (const QString &value, values)
            classifiers.append(QString("(SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=%1,outputVariable=predicted_%1)%2)").arg(key + "_" + value, isMeta ? QString("+Average+SaveMat(predicted_%1)").arg(value) : QString()));
        child = Transform::make(classifiers.join("/") + (classifiers.size() > 1 ? "+Cat" : ""));
    }

    void train(const QList<TemplateList> &data)
    {
        child->train(data);
    }

    void project(const Template &src, Template &dst) const
    {
        child->project(src, dst);
    }

    void store(QDataStream &stream) const
    {
        child->store(stream);
    }

    void load(QDataStream &stream)
    {
        child->load(stream);
    }
};

BR_REGISTER(Transform, TurkClassifierTransform)

} // namespace br

#include "classification/turk.moc"
