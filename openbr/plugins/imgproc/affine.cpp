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

#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Performs a two or three point registration.
 * \author Josh Klontz \cite jklontz
 * \note Method: Area should be used for shrinking an image, Cubic for slow but accurate enlargment, Bilin for fast enlargement.
 */
class AffineTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_ENUMS(Method)

public:
    /*!< */
    enum Method { Near = INTER_NEAREST,
                  Area = INTER_AREA,
                  Bilin = INTER_LINEAR,
                  Cubic = INTER_CUBIC,
                  Lanczo = INTER_LANCZOS4};

private:
    Q_PROPERTY(int width READ get_width WRITE set_width RESET reset_width STORED false)
    Q_PROPERTY(int height READ get_height WRITE set_height RESET reset_height STORED false)
    Q_PROPERTY(float x1 READ get_x1 WRITE set_x1 RESET reset_x1 STORED false)
    Q_PROPERTY(float y1 READ get_y1 WRITE set_y1 RESET reset_y1 STORED false)
    Q_PROPERTY(float x2 READ get_x2 WRITE set_x2 RESET reset_x2 STORED false)
    Q_PROPERTY(float y2 READ get_y2 WRITE set_y2 RESET reset_y2 STORED false)
    Q_PROPERTY(float x3 READ get_x3 WRITE set_x3 RESET reset_x3 STORED false)
    Q_PROPERTY(float y3 READ get_y3 WRITE set_y3 RESET reset_y3 STORED false)
    Q_PROPERTY(Method method READ get_method WRITE set_method RESET reset_method STORED false)
    Q_PROPERTY(bool storeAffine READ get_storeAffine WRITE set_storeAffine RESET reset_storeAffine STORED false)
    Q_PROPERTY(bool warpPoints READ get_warpPoints WRITE set_warpPoints RESET reset_warpPoints STORED false)
    BR_PROPERTY(int, width, 64)
    BR_PROPERTY(int, height, 64)
    BR_PROPERTY(float, x1, 0)
    BR_PROPERTY(float, y1, 0)
    BR_PROPERTY(float, x2, -1)
    BR_PROPERTY(float, y2, -1)
    BR_PROPERTY(float, x3, -1)
    BR_PROPERTY(float, y3, -1)
    BR_PROPERTY(Method, method, Bilin)
    BR_PROPERTY(bool, storeAffine, false)
    BR_PROPERTY(bool, warpPoints, false)

    static Point2f getThirdAffinePoint(const Point2f &a, const Point2f &b)
    {
        float dx = b.x - a.x;
        float dy = b.y - a.y;
        return Point2f(a.x - dy, a.y + dx);
    }

    void project(const Template &src, Template &dst) const
    {
        const bool twoPoints = ((x3 == -1) || (y3 == -1));

        Point2f dstPoints[3];
        dstPoints[0] = Point2f(x1*width, y1*height);
        dstPoints[1] = Point2f((x2 == -1 ? 1 - x1 : x2)*width, (y2 == -1 ? y1 : y2)*height);
        if (twoPoints) dstPoints[2] = getThirdAffinePoint(dstPoints[0], dstPoints[1]);
        else           dstPoints[2] = Point2f(x3*width, y3*height);

        Point2f srcPoints[3];
        if (src.file.contains("Affine_0") &&
            src.file.contains("Affine_1") &&
            (src.file.contains("Affine_2") || twoPoints)) {
            srcPoints[0] = OpenCVUtils::toPoint(src.file.get<QPointF>("Affine_0"));
            srcPoints[1] = OpenCVUtils::toPoint(src.file.get<QPointF>("Affine_1"));
            if (!twoPoints) srcPoints[2] = OpenCVUtils::toPoint(src.file.get<QPointF>("Affine_2"));
        } else {
            const QList<Point2f> landmarks = OpenCVUtils::toPoints(src.file.points());

            if ((landmarks.size() < 2) || (!twoPoints && (landmarks.size() < 3))) {
                resize(src, dst, Size(width, height));
                return;
            } else {
                srcPoints[0] = landmarks[0];
                srcPoints[1] = landmarks[1];
                if (!twoPoints) srcPoints[2] = landmarks[2];
            }
        }
        if (twoPoints) srcPoints[2] = getThirdAffinePoint(srcPoints[0], srcPoints[1]);

        Mat affineTransform = getAffineTransform(srcPoints, dstPoints);
        warpAffine(src, dst, affineTransform, Size(width, height), method);

        if (warpPoints) {
            QList<QPointF> points = src.file.points();
            QList<QPointF> rotatedPoints;
            for (int i=0; i<points.size(); i++) {
                rotatedPoints.append(QPointF(points.at(i).x()*affineTransform.at<double>(0,0)+
                                             points.at(i).y()*affineTransform.at<double>(0,1)+
                                             affineTransform.at<double>(0,2),
                                             points.at(i).x()*affineTransform.at<double>(1,0)+
                                             points.at(i).y()*affineTransform.at<double>(1,1)+
                                             affineTransform.at<double>(1,2)));
            }

            dst.file.setPoints(rotatedPoints);
        }

        if (storeAffine) {
            QList<float> affineParams;
            for (int i = 0 ; i < 2; i++)
                for (int j = 0; j < 3; j++)
                    affineParams.append(affineTransform.at<double>(i, j));
            dst.file.setList("affineParameters", affineParams);
        }
    }
};

BR_REGISTER(Transform, AffineTransform)

} // namespace br

#include "imgproc/affine.moc"
