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

#include <fstream>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>
#include <openbr/core/qtutils.h>

#include <opencv2/highgui/highgui.hpp>

using namespace cv;

namespace br
{
/*
class CascadeResourceMaker : public ResourceMaker<CascadeClassifier>
{
    QString file;

public:
    CascadeResourceMaker(const QString &model)
    {
        file = Globals->sdkPath + "/share/openbr/models/";
        if      (model == "Ear")         file += "haarcascades/haarcascade_ear.xml";
        else if (model == "Eye")         file += "haarcascades/haarcascade_eye_tree_eyeglasses.xml";
        else if (model == "FrontalFace") file += "haarcascades/haarcascade_frontalface_alt2.xml";
        else if (model == "ProfileFace") file += "haarcascades/haarcascade_profileface.xml";
        else {
            // Create a directory for trainable cascades
            file += "openbrcascades/"+model+"/cascade.xml";
            QFile touchFile(file);
            QtUtils::touchDir(touchFile);
        }
    }

private:
    CascadeClassifier *make() const
    {
        CascadeClassifier *cascade = new CascadeClassifier();
        if (!cascade->load(file.toStdString()))
            qFatal("Failed to load: %s", qPrintable(file));
        return cascade;
    }
};
*/
/*!
 * \ingroup transforms
 * \brief Applies a classifier to a sliding window.
 *        Discards negative detections.
 * \author Jordan Cheney \cite JordanCheney
 */
 /*
class SlidingWindowTransform : public Transform
{
    Q_OBJECT

    Q_PROPERTY(br::Classifier *classifier READ get_classifier WRITE set_classifier RESET reset_classifier STORED false)
    Q_PROPERTY(QString vecFile READ get_vecFile WRITE set_vecFile RESET reset_vecFile STORED false)
    Q_PROPERTY(QString negFile READ get_negFile WRITE set_negFile RESET reset_negFile STORED false)
    Q_PROPERTY(int minSize READ get_minSize WRITE set_minSize RESET reset_minSize STORED false)
    Q_PROPERTY(int maxSize READ get_maxSize WRITE set_maxSize RESET reset_maxSize STORED false)
    Q_PROPERTY(float scaleFactor READ get_scaleFactor WRITE set_scaleFactor RESET reset_scaleFactor STORED false)

    BR_PROPERTY(br::Classifier *, classifier, NULL)
    BR_PROPERTY(QString, vecFile, "vec.vec")
    BR_PROPERTY(QString, negFile, "neg.txt")
    BR_PROPERTY(int, minSize, 20)
    BR_PROPERTY(int, maxSize, -1)
    BR_PROPERTY(float, scaleFactor, 1.2)

    Resource<CascadeClassifier> cascadeResource;

    QList<Mat> getPos()
    {
        FILE *file = fopen(vecFile.toStdString().c_str(), "rb");
        if ( !file )
            qFatal("Couldn't open the file");

        short* vec = 0;
        int count, vecSize, last, base;

        short tmp = 0;
        if( fread( &count, sizeof( count ), 1, file ) != 1 ||
            fread( &vecSize, sizeof( vecSize ), 1, file ) != 1 ||
            fread( &tmp, sizeof( tmp ), 1, file ) != 1 ||
            fread( &tmp, sizeof( tmp ), 1, file ) != 1 )
            CV_Error_( CV_StsParseError, ("wrong file format for %s\n", qPrintable(vecFile)) );
        base = sizeof( count ) + sizeof( vecSize ) + 2*sizeof( tmp );

        last = 0;
        vec = (short*) cvAlloc( sizeof( *vec ) * vecSize );
        CV_Assert( vec );

        QList<Mat> posImages;
        for (int i = 0; i < 35770; i++) {
            Mat pos(24, 24, CV_8UC1);

            CV_Assert( pos.rows * pos.cols == vecSize );
            uchar tmp = 0;
            size_t elements_read = fread( &tmp, sizeof( tmp ), 1, file );
            if( elements_read != 1 )
                CV_Error( CV_StsBadArg, "Can not get new positive sample. The most possible reason is "
                                        "insufficient count of samples in given vec-file.\n");
            elements_read = fread( vec, sizeof( vec[0] ), vecSize, file );
            if( elements_read != (size_t)(vecSize) )
                CV_Error( CV_StsBadArg, "Can not get new positive sample. Seems that vec-file has incorrect structure.\n");

            if( feof( file ) || last++ >= count )
                CV_Error( CV_StsBadArg, "Can not get new positive sample. vec-file is over.\n");

            for( int r = 0; r < pos.rows; r++ )
                for( int c = 0; c < pos.cols; c++ )
                    pos.ptr(r)[c] = (uchar)vec[r * pos.cols + c];
            posImages.append(pos);
        }
        return posImages;
    }

    QList<Mat> getNeg()
    {
        QList<Mat> negs;

        std::string dirname, str;
        std::ifstream file(negFile.toStdString().c_str());

        size_t pos = negFile.toStdString().rfind('\\');
        char dlmrt = '\\';
        if (pos == string::npos)
        {
            pos = negFile.toStdString().rfind('/');
            dlmrt = '/';
        }
        dirname = pos == string::npos ? "" : negFile.toStdString().substr(0, pos) + dlmrt;
        while( !file.eof() )
        {
            std::getline(file, str);
            if (str.empty()) break;
            if (str.at(0) == '#' ) continue;
            negs.append(imread(dirname + str, CV_LOAD_IMAGE_GRAYSCALE));
        }
        file.close();

        return negs;
    }

    void init()
    {
        cascadeResource.setResourceMaker(new CascadeResourceMaker(model));
        if (model == "Ear" || model == "Eye" || model == "FrontalFace" || model == "ProfileFace")
            this->trainable = false;
    }

    void train(const TemplateList &_data)
    {
        (void)_data;

        QList<Mat> posImages = getPos();
        QList<Mat> negImages = getNeg();

        QList<Mat> images; QList<float> labels;
        foreach (const Mat &pos, posImages) {
            images.append(pos);
            labels.append(1.);
        }

        foreach (const Mat &neg, negImages) {
            images.append(neg);
            labels.append(0.);
        }

        classifier->train(images, labels);
    }

        void project(const Template &src, Template &dst) const
    {
        TemplateList temp;
        project(TemplateList() << src, temp);
        if (!temp.isEmpty()) dst = temp.first();
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        CascadeClassifier *cascade = cascadeResource.acquire();
        foreach (const Template &t, src) {
            const bool enrollAll = t.file.getBool("enrollAll");

            // Mirror the behavior of ExpandTransform in the special case
            // of an empty template.
            if (t.empty() && !enrollAll) {
                dst.append(t);
                continue;
            }

            for (int i=0; i<t.size(); i++) {
                Mat m;
                OpenCVUtils::cvtUChar(t[i], m);

                Size maxSize(m.cols, m.rows);

                Mat imageBuffer(m.rows + 1, m.cols + 1, CV_8U);

                QList<Rect> rects;
                QList<int> levels;
                QList<double> weights;

                for (double factor = 1; ; factor *= 1.2) {
                    Size originalWindowSize = Size(winWidth, winHeight);

                    Size windowSize( cvRound(originalWindowSize.width*factor), cvRound(originalWindowSize.height*factor) );
                    Size scaledImageSize( cvRound( m.cols/factor ), cvRound( m.rows/factor ) );
                    Size processingRectSize( scaledImageSize.width - originalWindowSize.width, scaledImageSize.height - originalWindowSize.height );

                    if( processingRectSize.width <= 0 || processingRectSize.height <= 0 )
                        break;
                    if( windowSize.width > maxSize.width || windowSize.height > maxSize.height )
                        break;
                    if( windowSize.width < minSize || windowSize.height < minSize )
                        continue;

                    Mat scaledImage( scaledImageSize, CV_8U, imageBuffer.data );
                    resize( m, scaledImage, scaledImageSize, 0, 0, CV_INTER_LINEAR );

                    int yStep = factor > 2. ? 1 : 2;
                    int stripCount, stripSize;

                    const int PTS_PER_THREAD = 1000;
                    stripCount = ((processingRectSize.width/yStep)*(processingRectSize.height + yStep-1)/yStep + PTS_PER_THREAD/2)/PTS_PER_THREAD;
                    stripCount = std::min(std::max(stripCount, 1), 100);
                    stripSize = (((processingRectSize.height + stripCount - 1)/stripCount + yStep-1)/yStep)*yStep;

                    if( !cascade->setImage( scaledImage ) )
                        qFatal("Can't set an image. Don't know why");

                    for( int y = 0; y < min(stripCount * stripSize, processingRectSize.height); y += yStep ) {
                        for( int x = 0; x < processingRectSize.width; x += yStep ) {
                            double gypWeight;
                            int result = cascade->runAt(cascade->featureEvaluator, Point(x, y), gypWeight);

                            if (ROCMode) {
                                if (result == 1)
                                    result =  -(int)cascade->data.stages.size();
                                if (cascade->data.stages.size() + result < 4 ) {
                                    rects.append(Rect(cvRound(x*factor), cvRound(y*factor), windowSize.width, windowSize.height));
                                    levels.append(-result);
                                    weights.append(gypWeight);
                                }
                            }
                            else if( result > 0 ) {
                                rects.append(Rect(cvRound(x*factor), cvRound(y*factor), windowSize.width, windowSize.height));
                            }
                            if( result == 0 )
                                x += yStep;
                        }
                    }
                }

                if (ROCMode)
                    groupRectangles(rects, levels, weights, minNeighbors, 0.2);
                else
                    groupRectangles(rects, minNeighbors, 0.2);

                if (!enrollAll && rects.empty())
                    rects.append(Rect(0, 0, m.cols, m.rows));

                for (int j = 0; j < rects.size(); j++) {
                    Template u(t.file, m);
                    if (levels.size() > j)
                        u.file.set("Confidence", levels[j]*weights[j]);
                    else
                        u.file.set("Confidence", 1);
                    const QRectF rect = OpenCVUtils::fromRect(rects[j]);
                    u.file.appendRect(rect);
                    u.file.set(model, rect);
                    dst.append(u);
                }
            }
        }
        cascadeResource.release(cascade);
    }
};

BR_REGISTER(Transform, SlidingWindowTransform)
*/

} // namespace br

#include "imgproc/slidingwindow.moc"
