/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2015 Rank One Computing 
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
 * \brief Randomly translate an image based on the height of a face contained within
 * \br_param maxHeight The max percentage of face height to shift the image
 * \author Brendan Klare \cite bklare
 */
class RndTranslateTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    Q_PROPERTY(float maxHeight READ get_maxHeight WRITE set_maxHeight RESET reset_maxHeight STORED false)
    BR_PROPERTY(float, maxHeight, .1)
    Q_PROPERTY(int nStages READ get_nStages WRITE set_nStages RESET reset_nStages STORED false)
    BR_PROPERTY(int, nStages, 3)

    void project(const Template &src, Template &dst) const 
    {
        qFatal("Shoult not  be here (RndTranslate)");
    }

    void project(const TemplateList &srcList, TemplateList &dstList) const
    {
        foreach (const Template &src, srcList) {
            for (int stage = 0; stage < nStages; stage++) { 

                Template dst = src;

                QPointF rightEye = src.file.get<QPointF>("RightEye");
                QPointF leftEye = src.file.get<QPointF>("LeftEye");
                QPointF chin = src.file.get<QPointF>("Chin");
                QPointF eyeCenter = (rightEye + leftEye) / 2;
                const float length = sqrt(pow(eyeCenter.x() - chin.x(), 2.0) +
                                          pow(eyeCenter.y() - chin.y(), 2.0));

                int max = qRound(length * maxHeight);
                int shiftX = (rand() % (max * 2 + 1)) - max;
                int shiftY = (rand() % (max * 2 + 1)) - max;
                //Mat out(src.m().rows, src.m().cols, src.m().type());
                Mat out;
                Mat M = Mat::zeros(2, 3, CV_32F);
                M.at<float>(0,0) = 1;
                M.at<float>(1,1) = 1;
                M.at<float>(0,2) = shiftX;
                M.at<float>(1,2) = shiftY;
                warpAffine(src.m(), out, M, Size(src.m().rows, src.m().cols));

                dst.m() = out;
                dstList += dst;
            }
        }
    }
};

BR_REGISTER(Transform, RndTranslateTransform)

} // namespace br

#include "imgproc/rndtranslate.moc"
