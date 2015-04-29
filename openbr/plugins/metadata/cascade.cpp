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
#include <QProcess>
#include <QTemporaryFile>
#include <opencv2/objdetect/objdetect.hpp>
#include <fstream>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>
#include <openbr/core/resource.h>
#include <openbr/core/qtutils.h>
#include <openbr/core/cascade.h>

using namespace cv;

namespace br
{
        
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

/*!
 * \ingroup transforms
 * \brief Wraps OpenCV cascade classifier
 * \author Josh Klontz \cite jklontz
 * \author David Crouse \cite dgcrouse
 */
class CascadeTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString model READ get_model WRITE set_model RESET reset_model STORED false)
    Q_PROPERTY(int minSize READ get_minSize WRITE set_minSize RESET reset_minSize STORED false)
    Q_PROPERTY(int minNeighbors READ get_minNeighbors WRITE set_minNeighbors RESET reset_minNeighbors STORED false)
    Q_PROPERTY(bool ROCMode READ get_ROCMode WRITE set_ROCMode RESET reset_ROCMode STORED false)
    
    // Training parameters
    Q_PROPERTY(QString vecFile READ get_vecFile WRITE set_vecFile RESET reset_vecFile STORED false)
    Q_PROPERTY(QString negFile READ get_negFile WRITE set_negFile RESET reset_negFile STORED false)
    Q_PROPERTY(int winWidth READ get_winWidth WRITE set_winWidth RESET reset_winWidth STORED false)
    Q_PROPERTY(int winHeight READ get_winHeight WRITE set_winHeight RESET reset_winHeight STORED false)
    Q_PROPERTY(int numPos READ get_numPos WRITE set_numPos RESET reset_numPos STORED false)
    Q_PROPERTY(int numNeg READ get_numNeg WRITE set_numNeg RESET reset_numNeg STORED false)
    Q_PROPERTY(int numStages READ get_numStages WRITE set_numStages RESET reset_numStages STORED false)

    BR_PROPERTY(QString, model, "FrontalFace")
    BR_PROPERTY(int, minSize, 64)
    BR_PROPERTY(int, minNeighbors, 5)
    BR_PROPERTY(bool, ROCMode, false)                 

    BR_PROPERTY(QString, vecFile, "vec.vec")
    BR_PROPERTY(QString, negFile, "neg.txt")
    BR_PROPERTY(int, winWidth, 24)
    BR_PROPERTY(int, winHeight, 24)
    BR_PROPERTY(int, numPos, 1000)
    BR_PROPERTY(int, numNeg, 1000)
    BR_PROPERTY(int, numStages, 20)

    Resource<CascadeClassifier> cascadeResource;

    void init()
    {
        cascadeResource.setResourceMaker(new CascadeResourceMaker(model));
        if (model == "Ear" || model == "Eye" || model == "FrontalFace" || model == "ProfileFace")
            this->trainable = false;
    }
    
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
            Mat pos(winHeight, winWidth, CV_8UC1);

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

    // Train transform
    void train(const TemplateList& data)
    {
        (void) data;

        QList<Mat> posImages = getPos();
        QList<Mat> negImages = getNeg();

        BrCascadeClassifier classifier;

        CascadeBoostParams stageParams(CvBoost::GENTLE, 0.999, 0.5, 0.95, 1, 200);

        Representation *representation = Representation::make("MBLBP(24,24)", NULL);

        QString cascadeDir = Globals->sdkPath + "/share/openbr/models/openbrcascades/" + model;
        classifier.train(cascadeDir.toStdString(),
                         posImages, negImages,
                         1024, 1024, numPos, numNeg, numStages,
                         representation, stageParams);
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
                std::vector<Rect> rects;
                std::vector<int> rejectLevels;
                std::vector<double> levelWeights;
                if (ROCMode) cascade->detectMultiScale(m, rects, rejectLevels, levelWeights, 1.2, minNeighbors, (enrollAll ? 0 : CASCADE_FIND_BIGGEST_OBJECT) | CASCADE_SCALE_IMAGE, Size(minSize, minSize), Size(), true);
                else         cascade->detectMultiScale(m, rects, 1.2, minNeighbors, enrollAll ? 0 : CASCADE_FIND_BIGGEST_OBJECT, Size(minSize, minSize));

                if (!enrollAll && rects.empty())
                    rects.push_back(Rect(0, 0, m.cols, m.rows));

                for (size_t j=0; j<rects.size(); j++) {
                    Template u(t.file, m);
                    if (rejectLevels.size() > j)
                        u.file.set("Confidence", rejectLevels[j]*levelWeights[j]);
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

    // TODO: Remove this code when ready to break binary compatibility
    void store(QDataStream &stream) const
    {
        int size = 1;
        stream << size;
    }

    void load(QDataStream &stream)
    {
        int size;
        stream >> size;
    }
};

BR_REGISTER(Transform, CascadeTransform)

} // namespace br

#include "metadata/cascade.moc"
