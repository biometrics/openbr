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

#include <QCryptographicHash>

#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Wraps QCryptographicHash
 * \br_link http://doc.qt.io/qt-5/qcryptographichash.html
 * \author Josh Klontz \cite jklontz
 */
class CryptographicHashTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_ENUMS(Algorithm)
    Q_PROPERTY(Algorithm algorithm READ get_algorithm WRITE set_algorithm RESET reset_algorithm STORED false)

public:
    /*!< */
    enum Algorithm { Md4 = QCryptographicHash::Md4,
                     Md5 = QCryptographicHash::Md5,
                     Sha1 = QCryptographicHash::Sha1 };

private:
    BR_PROPERTY(Algorithm, algorithm, Md5)

    void project(const Template &src, Template &dst) const
    {
        const cv::Mat &m = src;
        const QByteArray data((const char *)m.data, int(m.total()*m.elemSize()));
        const QByteArray hash = QCryptographicHash::hash(data, (QCryptographicHash::Algorithm)algorithm);
        cv::Mat n(1, hash.size(), CV_8UC1);
        memcpy(n.data, hash.constData(), hash.size());
        dst = Template(src.file, n);
    }
};

BR_REGISTER(Transform, CryptographicHashTransform)

} // namespace br

#include "imgproc/cryptographichash.moc"
