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
#include "openbr_internal.h"

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV inpainting
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

BR_REGISTER(Transform, InpaintTransform)

/*!
 * \ingroup transforms
 * \brief Fill 0 pixels with the mean of non-0 pixels.
 * \author Josh Klontz \cite jklontz
 */
class MeanFillTransform : public UntrainableTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst = src.m().clone();
        dst.m().setTo(mean(dst, dst.m()!=0), dst.m()==0);
    }
};

BR_REGISTER(Transform, MeanFillTransform)

/*!
 * \ingroup transforms
 * \brief Fill black pixels with the specified color.
 * \author Josh Klontz \cite jklontz
 */
class FloodTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int r READ get_r WRITE set_r RESET reset_r STORED false)
    Q_PROPERTY(int g READ get_g WRITE set_g RESET reset_g STORED false)
    Q_PROPERTY(int b READ get_b WRITE set_b RESET reset_b STORED false)
    Q_PROPERTY(bool all READ get_all WRITE set_all RESET reset_all STORED false)
    BR_PROPERTY(int, r, 0)
    BR_PROPERTY(int, g, 0)
    BR_PROPERTY(int, b, 0)
    BR_PROPERTY(bool, all, false)

    void project(const Template &src, Template &dst) const
    {
        dst = src.m().clone();
        dst.m().setTo(Scalar(r, g, b), all ? Mat() : dst.m()==0);
    }
};

BR_REGISTER(Transform, FloodTransform)

/*!
 * \ingroup transforms
 * \brief Alpha-blend two matrices
 * \author Josh Klontz \cite jklontz
 */
class BlendTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(float alpha READ get_alpha WRITE set_alpha RESET reset_alpha STORED false)
    BR_PROPERTY(float, alpha, 0.5)

    void project(const Template &src, Template &dst) const
    {
        if (src.size() != 2) qFatal("Expected two source matrices.");
        addWeighted(src[0], alpha, src[1], 1-alpha, 0, dst);
    }
};

BR_REGISTER(Transform, BlendTransform)

} // namespace br

#include "fill.moc"
