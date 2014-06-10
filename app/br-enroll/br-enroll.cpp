/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2014 Noblis                                                     *
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

#include <QtCore>
#include <opencv2/highgui/highgui.hpp>
#include <cstdio>
#include <cstring>
#include <openbr/openbr_plugin.h>
#include <openbr/universal_template.h>

using namespace br;
using namespace cv;

static void help()
{
    printf("br-enroll [args]\n"
           "================\n"
           "* __stdin__  - Templates (raw data)\n"
           "* __stdout__ - Templates (feature vectors)\n"
           "\n"
           "_br-enroll_ is an application that creates feature vector template(s) from images.\n"
           "For every input image template in _stdin_, enroll writes zero or more templates to _stdout_.\n"
           "\n"
           "Enroll may choose to store metadata in the feature vector in an algorithm-specific manner.\n"
           "For example, a face recognition algorithm may devote the first 16-bytes of the feature vector to saving four 4-byte integers representing the face bounding box (X, Y, Width, Height).\n"
           "It is expected that _br-search_ will understand and output algorithm-specific metadata for the top matching templates.\n");
}

static QSharedPointer<Transform> algorithm;

static void enroll_utemplate(br_const_utemplate utemplate, br_callback_context)
{
    if (utemplate->algorithmID != 3)
        qFatal("Expected an encoded image.");

    TemplateList templates;
    templates.append(Template(imdecode(Mat(1, utemplate->size, CV_8UC1, (void*) utemplate->data), IMREAD_UNCHANGED)));
    templates >> *algorithm;

    foreach (const Template &t, templates) {
        const Mat &m = t.m();
        QByteArray data((const char*) m.data, m.rows * m.cols * m.elemSize());

        const QRectF frontalFace = t.file.get<QRectF>("FrontalFace");
        const QPointF firstEye   = t.file.get<QPointF>("First_Eye");
        const QPointF secondEye  = t.file.get<QPointF>("Second_Eye");
        const float x         = frontalFace.x();
        const float y         = frontalFace.y();
        const float width     = frontalFace.width();
        const float height    = frontalFace.height();
        const float rightEyeX = firstEye.x();
        const float rightEyeY = firstEye.y();
        const float leftEyeX  = secondEye.x();
        const float leftEyeY  = secondEye.y();

        data.append((const char*)&x        , sizeof(float));
        data.append((const char*)&y        , sizeof(float));
        data.append((const char*)&width    , sizeof(float));
        data.append((const char*)&height   , sizeof(float));
        data.append((const char*)&rightEyeX, sizeof(float));
        data.append((const char*)&rightEyeY, sizeof(float));
        data.append((const char*)&leftEyeX , sizeof(float));
        data.append((const char*)&leftEyeY , sizeof(float));

        const QByteArray templateID = QCryptographicHash::hash(data, QCryptographicHash::Md5);
        br_append_utemplate_contents(stdout, utemplate->imageID, (const unsigned char*) templateID.data(), -1, data.size(), (const unsigned char*) data.data());
    }
}

int main(int argc, char *argv[])
{
    for (int i=1; i<argc; i++)
        if (!strcmp(argv[i], "-help")) { help(); exit(EXIT_SUCCESS); }

    Context::initialize(argc, argv, "", false);
    Globals->quiet = true;
    Globals->enrollAll = true;
    algorithm = Transform::fromAlgorithm("FaceRecognition");
    br_iterate_utemplates_file(stdin, enroll_utemplate, NULL);
    Context::finalize();
    return EXIT_SUCCESS;
}
