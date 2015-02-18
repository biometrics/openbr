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

#include <Eigen/Dense>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Sliding window feature extraction from a multi-channel integral image.
 * \author Josh Klontz \cite jklontz
 */
class IntegralSamplerTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(int scales READ get_scales WRITE set_scales RESET reset_scales STORED false)
    Q_PROPERTY(float scaleFactor READ get_scaleFactor WRITE set_scaleFactor RESET reset_scaleFactor STORED false)
    Q_PROPERTY(float stepFactor READ get_stepFactor WRITE set_stepFactor RESET reset_stepFactor STORED false)
    Q_PROPERTY(int minSize READ get_minSize WRITE set_minSize RESET reset_minSize STORED false)
    Q_PROPERTY(bool secondOrder READ get_secondOrder WRITE set_secondOrder RESET reset_secondOrder STORED false)
    BR_PROPERTY(int, scales, 6)
    BR_PROPERTY(float, scaleFactor, 2)
    BR_PROPERTY(float, stepFactor, 0.75)
    BR_PROPERTY(int, minSize, 8)
    BR_PROPERTY(bool, secondOrder, false)

    void project(const Template &src, Template &dst) const
    {
        typedef Eigen::Map< const Eigen::Matrix<qint32,Eigen::Dynamic,1> > InputDescriptor;
        typedef Eigen::Map< const Eigen::Matrix<float,Eigen::Dynamic,1> > SecondOrderInputDescriptor;
        typedef Eigen::Map< Eigen::Matrix<float,Eigen::Dynamic,1> > OutputDescriptor;

        const Mat &m = src.m();
        if (m.depth() != CV_32S) qFatal("Expected CV_32S matrix depth.");
        const int channels = m.channels();
        const int rowStep = channels * m.cols;

        int descriptors = 0;
        float idealSize = min(m.rows, m.cols)-1;
        for (int scale=0; scale<scales; scale++) {
            const int currentSize(idealSize);
            const int numDown = 1+(m.rows-currentSize-1)/int(idealSize*stepFactor);
            const int numAcross = 1+(m.cols-currentSize-1)/int(idealSize*stepFactor);
            descriptors += numDown*numAcross;
            if (secondOrder) descriptors += numDown*(numAcross-1) + (numDown-1)*numAcross;
            idealSize /= scaleFactor;
            if (idealSize < minSize) break;
        }
        Mat n(descriptors, channels, CV_32FC1);

        const qint32 *dataIn = (qint32*)m.data;
        float *dataOut = (float*)n.data;
        idealSize = min(m.rows, m.cols)-1;
        int index = 0;
        for (int scale=0; scale<scales; scale++) {
            const int currentSize(idealSize);
            const int currentStep(idealSize*stepFactor);
            for (int i=currentSize; i<m.rows; i+=currentStep) {
                for (int j=currentSize; j<m.cols; j+=currentStep) {
                    InputDescriptor a(dataIn+((i-currentSize)*rowStep+(j-currentSize)*channels), channels, 1);
                    InputDescriptor b(dataIn+((i-currentSize)*rowStep+ j             *channels), channels, 1);
                    InputDescriptor c(dataIn+(i              *rowStep+(j-currentSize)*channels), channels, 1);
                    InputDescriptor d(dataIn+(i              *rowStep+ j             *channels), channels, 1);
                    OutputDescriptor y(dataOut+(index*channels), channels, 1);
                    y = (d-b-c+a).cast<float>()/(currentSize*currentSize);
                    index++;
                }
            }
            if (secondOrder) {
                const int numDown = 1+(m.rows-currentSize-1)/currentStep;
                const int numAcross = 1+(m.cols-currentSize-1)/currentStep;
                const float *dataIn = n.ptr<float>(index - numDown*numAcross);
                for (int i=0; i<numDown; i++) {
                    for (int j=0; j<numAcross; j++) {
                        SecondOrderInputDescriptor a(dataIn + (i*numAcross+j)*channels, channels, 1);
                        if (j < numAcross-1) {
                            OutputDescriptor y(dataOut+(index*channels), channels, 1);
                            y = a - SecondOrderInputDescriptor(dataIn + (i*numAcross+j+1)*channels, channels, 1);
                            index++;
                        }
                        if (i < numDown-1) {
                            OutputDescriptor y(dataOut+(index*channels), channels, 1);
                            y = a - SecondOrderInputDescriptor(dataIn + ((i+1)*numAcross+j)*channels, channels, 1);
                            index++;
                        }
                    }
                }
            }
            idealSize /= scaleFactor;
            if (idealSize < minSize) break;
        }

        if (descriptors != index)
            qFatal("Allocated %d descriptors but computed %d.", descriptors, index);

        dst.m() = n;
    }
};

BR_REGISTER(Transform, IntegralSamplerTransform)

} // namespace br

#include "imgproc/integralsampler.moc"
