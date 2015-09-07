/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2015 Rank One Computing
 *                                                                           *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <opencv2/imgproc/imgproc.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Scales the mat values by provided factor
 * \author Brendan Klare \cite bklare
 */
class ScaleMatTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(float scaleFactor READ get_scaleFactor WRITE set_scaleFactor RESET reset_scaleFactor STORED false)
    BR_PROPERTY(float, scaleFactor, 1.)

    void project(const Template &src, Template &dst) const
    {
        dst = src * scaleFactor;
    }
};

BR_REGISTER(Transform, ScaleMatTransform)

} // namespace br

#include "imgproc/scalemat.moc"
