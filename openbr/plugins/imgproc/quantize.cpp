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
 * \brief Approximate floats as uchar.
 * \author Josh Klontz \cite jklontz
 */
class QuantizeTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(float a READ get_a WRITE set_a RESET reset_a)
    Q_PROPERTY(float b READ get_b WRITE set_b RESET reset_b)
    BR_PROPERTY(float, a, 1)
    BR_PROPERTY(float, b, 0)

    void train(const TemplateList &data)
    {
        double minVal, maxVal;
        minMaxLoc(OpenCVUtils::toMat(data.data()), &minVal, &maxVal);
        a = 255.0/(maxVal-minVal);
        b = -a*minVal;
        qDebug() << "Quantized dimensions =" << data.first().m().rows * data.first().m().cols;
    }

    void project(const Template &src, Template &dst) const
    {
        src.m().convertTo(dst, CV_8U, a, b);
    }

    QByteArray likely(const QByteArray &indentation) const
    {
        QByteArray result;
        result.append("\n" + indentation + "{ ; Quantize\n");
        result.append(indentation + "  dst := (imitate-size src (imitate-dimensions u8 src.type))\n");
        result.append(indentation + "  (dst src) :=>\n");
        result.append(indentation + "    dst :<- src.(* ");
        result.append(QByteArray::number(a, 'g', 9));
        result.append(").(+ ");
        result.append(QByteArray::number(b, 'g', 9));
        result.append(")\n");
        result.append(indentation + "}");
        return result;
    }
};

BR_REGISTER(Transform, QuantizeTransform)

} // namespace br

#include "imgproc/quantize.moc"
