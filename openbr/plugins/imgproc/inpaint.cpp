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

#include <opencv2/photo/photo.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV inpainting
 * \br_link http://docs.opencv.org/modules/photo/doc/inpainting.html
 * \author Josh Klontz \cite jklontz
 */
class InpaintTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_ENUMS(Method)
    Q_PROPERTY(int radius READ get_radius WRITE set_radius RESET reset_radius STORED false)
    Q_PROPERTY(Method method READ get_method WRITE set_method RESET reset_method STORED false)

public:
    /*!< */
    enum Method { NavierStokes = INPAINT_NS,
                  Telea = INPAINT_TELEA };

private:
    BR_PROPERTY(int, radius, 1)
    BR_PROPERTY(Method, method, NavierStokes)
    Transform *cvtGray;

    void init()
    {
        cvtGray = make("Cvt(Gray)");
    }

    void project(const Template &src, Template &dst) const
    {
        inpaint(src, (*cvtGray)(src)<5, dst, radius, method);
    }
};

} // namespace br

#include "imgproc/inpaint.moc"
