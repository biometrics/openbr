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

static void enroll_utemplate(br_const_utemplate utemplate)
{
    if (utemplate->algorithmID != 3)
        qFatal("Expected an encoded image.");

    TemplateList templates;
    templates.append(Template(imdecode(Mat(1, utemplate->size, CV_8UC1, (void*) utemplate->data), IMREAD_UNCHANGED)));
    templates >> *algorithm;

    foreach (const Template &t, templates) {
        const Mat &m = t.m();
        const uint32_t size = m.rows * m.cols * m.elemSize();
        const QByteArray templateID = QCryptographicHash::hash(QByteArray((const char*) m.data, size), QCryptographicHash::Md5);
        br_append_utemplate_contents(stdout, utemplate->imageID, (const int8_t*) templateID.data(), -1, size, (const int8_t*) m.data);
    }
}

int main(int argc, char *argv[])
{
    for (int i=1; i<argc; i++)
        if (!strcmp(argv[i], "-help")) { help(); exit(EXIT_SUCCESS); }

    Context::initialize(argc, argv, "", false);
    Globals->quiet = true;
    algorithm = Transform::fromAlgorithm("FaceRecognition");
    br_iterate_utemplates_file(stdin, enroll_utemplate);
    Context::finalize();
    return EXIT_SUCCESS;
}
