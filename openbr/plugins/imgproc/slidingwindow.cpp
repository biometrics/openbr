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

/*!
 * \ingroup transforms
 * \brief Applies a classifier to a sliding window.
 *        Discards negative detections.
 * \author Jordan Cheney \cite JordanCheney
 */

class SlidingWindowTransform : public Transform
{
    Q_OBJECT

    Q_PROPERTY(br::Classifier *classifier READ get_classifier WRITE set_classifier RESET reset_classifier STORED false)
    Q_PROPERTY(QString cascadeDir READ get_cascadeDir WRITE set_cascadeDir RESET reset_cascadeDir STORED false)
    Q_PROPERTY(QString vecFile READ get_vecFile WRITE set_vecFile RESET reset_vecFile STORED false)
    Q_PROPERTY(QString negFile READ get_negFile WRITE set_negFile RESET reset_negFile STORED false)

    BR_PROPERTY(br::Classifier *, classifier, NULL)
    BR_PROPERTY(QString, cascadeDir, "")
    BR_PROPERTY(QString, vecFile, "vec.vec")
    BR_PROPERTY(QString, negFile, "neg.txt")

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
        (void)src; (void)dst;
    }

    void load(QDataStream &stream)
    {
        (void) stream;
        return;
    }

    void store(QDataStream &stream) const
    {
        (void) stream;

	QString path = Globals->sdkPath + "/share/openbr/models/openbrcascades/" + cascadeDir;
	QtUtils::touchDir(QDir(path));

	QString filename = path + "/cascade.xml";
        FileStorage fs(filename.toStdString(), FileStorage::WRITE);

        if (!fs.isOpened()) {
	    qWarning("Unable to open file: %s", qPrintable(filename));
            return;
	}
	
        fs << FileStorage::getDefaultObjectName(filename.toStdString()) << "{";

        classifier->write(fs);

        fs << "}";
    }
};

BR_REGISTER(Transform, SlidingWindowTransform)

} // namespace br

#include "imgproc/slidingwindow.moc"
