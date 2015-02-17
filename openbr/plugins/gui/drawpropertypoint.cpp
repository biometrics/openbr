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
 * \brief Draw the value of the specified property at the specified point on the image
 *
 * The inPlace argument controls whether or not the image is cloned before it is drawn on.
 *
 * \author Charles Otto \cite caotto
 */
class DrawPropertyPointTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(QString propName READ get_propName WRITE set_propName RESET reset_propName STORED false)
    Q_PROPERTY(QString pointName READ get_pointName WRITE set_pointName RESET reset_pointName STORED false)
    Q_PROPERTY(bool inPlace READ get_inPlace WRITE set_inPlace RESET reset_inPlace STORED false)
    BR_PROPERTY(QString, propName, "")
    BR_PROPERTY(QString, pointName, "")
    BR_PROPERTY(bool, inPlace, false)


    void project(const Template &src, Template &dst) const
    {
        dst = src;
        if (propName.isEmpty() || pointName.isEmpty())
            return;

        dst.m() = inPlace ? src.m() : src.m().clone();

        const Scalar textColor(255, 255, 0);

        QVariant prop = dst.file.value(propName);


        if (!prop.canConvert(QVariant::String))
            return;
        QString propString = prop.toString();

        QVariant point = dst.file.value(pointName);

        if (!point.canConvert(QVariant::PointF))
            return;

        QPointF targetPoint = point.toPointF();

        Point2f cvPoint = OpenCVUtils::toPoint(targetPoint);

        std::string text = propName.toStdString() + ": " + propString.toStdString();
        putText(dst, text, cvPoint, FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1);
    }

};
BR_REGISTER(Transform, DrawPropertyPointTransform)

} // namespace br

#include "gui/drawpropertypoint.moc"
