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

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Implements a Gabor Filter
 * \br_link http://en.wikipedia.org/wiki/Gabor_filter
 * \author Josh Klontz \cite jklontz
 */
class GaborTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_ENUMS(Component)
    Q_PROPERTY(float lambda READ get_lambda WRITE set_lambda RESET reset_lambda STORED false)
    Q_PROPERTY(float theta READ get_theta WRITE set_theta RESET reset_theta STORED false)
    Q_PROPERTY(float psi READ get_psi WRITE set_psi RESET reset_psi STORED false)
    Q_PROPERTY(float sigma READ get_sigma WRITE set_sigma RESET reset_sigma STORED false)
    Q_PROPERTY(float gamma READ get_gamma WRITE set_gamma RESET reset_gamma STORED false)
    Q_PROPERTY(Component component READ get_component WRITE set_component RESET reset_component STORED false)

public:
    /*!< */
    enum Component { Real,
                     Imaginary,
                     Magnitude,
                     Phase };

private:
    BR_PROPERTY(float, lambda, 0)
    BR_PROPERTY(float, theta, 0)
    BR_PROPERTY(float, psi, 0)
    BR_PROPERTY(float, sigma, 0)
    BR_PROPERTY(float, gamma, 0)
    BR_PROPERTY(Component, component, Phase)

    Mat kReal, kImaginary;

    friend class GaborJetTransform;

    static void makeWavelet(float lambda, float theta, float psi, float sigma, float gamma, Mat &kReal, Mat &kImaginary)
    {
        float sigma_x = sigma;
        float sigma_y = sigma/gamma;

        // Bounding box
        const double nstds = 3;
        int xmax = std::ceil(std::max(1.0, std::max(std::abs(nstds*sigma_x*cos(theta)),
                                                    std::abs(nstds*sigma_y*sin(theta)))));
        int ymax = std::ceil(std::max(1.0, std::max(std::abs(nstds*sigma_x*sin(theta)),
                                                    std::abs(nstds*sigma_y*cos(theta)))));

        // Compute kernels
        kReal.create(2*ymax+1, 2*xmax+1, CV_32FC1);
        kImaginary.create(2*ymax+1, 2*xmax+1, CV_32FC1);
        for (int y = -ymax; y <= ymax; y++) {
            int row = y + ymax;
            for (int x = -xmax; x <= xmax; x++) {
                int col = x + xmax;
                float x_prime = x*cos(theta) + y*sin(theta);
                float y_prime = -x*sin(theta) + y*cos(theta);
                float a = exp(-0.5 * (x_prime*x_prime + gamma*gamma*y_prime*y_prime)/(sigma*sigma));
                float b = 2*CV_PI*x_prime/lambda+psi;
                kReal.at<float>(row, col) = a*cos(b);
                kImaginary.at<float>(row, col) = a*sin(b);
            }
        }

        // Remove DC component, should only effect real kernel
        subtract(kReal, mean(kReal), kReal);
        subtract(kImaginary, mean(kImaginary), kImaginary);
    }

    void init()
    {
        makeWavelet(lambda, theta, psi, sigma, gamma, kReal, kImaginary);
    }

    void project(const Template &src, Template &dst) const
    {
        Mat real, imaginary, magnitude, phase;
        if (component != Imaginary)
            filter2D(src, real, -1, kReal);
        if (component != Real)
            filter2D(src, imaginary, -1, kImaginary);
        if ((component == Magnitude) || (component == Phase))
            cartToPolar(real, imaginary, magnitude, phase);

        if      (component == Real)      dst = real;
        else if (component == Imaginary) dst = imaginary;
        else if (component == Magnitude) dst = magnitude;
        else if (component == Phase)     dst = phase;
        else                             qFatal("Invalid component.");
    }
};

BR_REGISTER(Transform, GaborTransform)

/*!
 * \ingroup transforms
 * \brief A vector of gabor wavelets applied at a point.
 * \author Josh Klontz \cite jklontz
 */
class GaborJetTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_ENUMS(br::GaborTransform::Component)
    Q_PROPERTY(QList<float> lambdas READ get_lambdas WRITE set_lambdas RESET reset_lambdas STORED false)
    Q_PROPERTY(QList<float> thetas READ get_thetas WRITE set_thetas RESET reset_thetas STORED false)
    Q_PROPERTY(QList<float> psis READ get_psis WRITE set_psis RESET reset_psis STORED false)
    Q_PROPERTY(QList<float> sigmas READ get_sigmas WRITE set_sigmas RESET reset_sigmas STORED false)
    Q_PROPERTY(QList<float> gammas READ get_gammas WRITE set_gammas RESET reset_gammas STORED false)
    Q_PROPERTY(br::GaborTransform::Component component READ get_component WRITE set_component RESET reset_component STORED false)
    BR_PROPERTY(QList<float>, lambdas, QList<float>())
    BR_PROPERTY(QList<float>, thetas, QList<float>())
    BR_PROPERTY(QList<float>, psis, QList<float>())
    BR_PROPERTY(QList<float>, sigmas, QList<float>())
    BR_PROPERTY(QList<float>, gammas, QList<float>())
    BR_PROPERTY(GaborTransform::Component, component, GaborTransform::Phase)

    QList<Mat> kReals, kImaginaries;

    void init()
    {
        kReals.clear();
        kImaginaries.clear();
        foreach (float lambda, lambdas)
            foreach (float theta, thetas)
                foreach (float psi, psis)
                    foreach (float sigma, sigmas)
                        foreach (float gamma, gammas) {
                            Mat kReal, kImaginary;
                            GaborTransform::makeWavelet(lambda, theta, psi, sigma, gamma, kReal, kImaginary);
                            kReals.append(kReal);
                            kImaginaries.append(kImaginary);
                        }
    }

    static float response(const cv::Mat &src, const QPointF &point, const Mat &kReal, const Mat &kImaginary, GaborTransform::Component component)
    {
        Rect roi(std::max(std::min((int)(point.x() - kReal.cols/2.f), src.cols - kReal.cols), 0),
                 std::max(std::min((int)(point.y() - kReal.rows/2.f), src.rows - kReal.rows), 0),
                 kReal.cols,
                 kReal.rows);

        float real = 0, imaginary = 0, magnitude = 0, phase = 0;
        if (component != GaborTransform::Imaginary) {
            Mat dst;
            multiply(src(roi), kReal, dst);
            real = sum(dst)[0];
        }
        if (component != GaborTransform::Real) {
            Mat dst;
            multiply(src(roi), kImaginary, dst);
            imaginary = sum(dst)[0];
        }
        if ((component == GaborTransform::Magnitude) || (component == GaborTransform::Phase)) {
            magnitude = sqrt(real*real + imaginary*imaginary);
            phase = atan2(imaginary, real)*180/CV_PI;
        }

        float dst = 0;
        if      (component == GaborTransform::Real)      dst = real;
        else if (component == GaborTransform::Imaginary) dst = imaginary;
        else if (component == GaborTransform::Magnitude) dst = magnitude;
        else if (component == GaborTransform::Phase)     dst = phase;
        else                                             qFatal("Invalid component.");
        return dst;
    }

    void project(const Template &src, Template &dst) const
    {
        const QList<QPointF> points = src.file.points();
        dst = Mat(points.size(), kReals.size(), CV_32FC1);
        for (int i=0; i<points.size(); i++)
            for (int j=0; j<kReals.size(); j++)
                    dst.m().at<float>(i,j) = response(src, points[i], kReals[j], kImaginaries[j], component);
    }
};

BR_REGISTER(Transform, GaborJetTransform)

} // namespace br

#include "imgproc/gabor.moc"
