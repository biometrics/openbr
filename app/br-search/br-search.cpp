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
#include <cstdio>
#include <cstring>
#include <limits>
#include <openbr/openbr_plugin.h>
#include <openbr/universal_template.h>

using namespace br;
using namespace cv;

static void help()
{
    printf("br-search URL(s) [args]\n"
           "=======================\n"
           "* __stdin__  - Templates (feature vectors)\n"
           "* __stdout__ - JSON\n"
           "\n"
           "_br-search_ does retrieval by comparing query templates to target gallery(s).\n"
           "The search strategy is implementation defined.\n"
           "\n"
           "For every template read from _stdin_, search writes the top sorted matches as JSON objects to _stdout_ by comparing the query template against gallery URLs.\n"
           "The JSON objects include `AlgorithmID`, `QueryImageID`, `QueryTemplateID`, `TargetImageID`, `TargetTemplateID`, `Score`, and any algorithm-specific metadata fields set during _enroll_. \n"
           "\n"
           "Optional Arguments\n"
           "------------------\n"
           "* -help        - Print usage information.\n"
           "* -limit <int> - Maximum number of returns (20 otherwise).\n");
}

static int limit = 20;
static float threshold = -std::numeric_limits<float>::max();

struct Result
{
    int8_t targetImageID[16], targetTemplateID[16], queryImageID[16], queryTemplateID[16];
    int32_t algorithmID;
    float score;
};

struct TopTargets : QList< QPair<br_const_utemplate, float> >
{
    br_const_utemplate query;

    TopTargets(br_const_utemplate query)
        : query(query) {}

    void tryAdd(br_const_utemplate target, float score)
    {
        if ((score < threshold) || ((size() == limit) && (score < last().second)))
            return;
        (void) target;
    }

    void print() const
    {

    }
};

struct FaceRecognitionResult : public Result
{
    int32_t x, y, width, height;

    FaceRecognitionResult()
    {
        algorithmID = -1;
    }
};

struct MappedGallery
{
    QSharedPointer<QFile> file;
    qint64 size;
    uchar *data;

    MappedGallery(QString url)
    {
        if (url.startsWith("file://"))
            url = url.mid(7);
        file.reset(new QFile(url));
        file->open(QFile::ReadOnly);
        size = file->size();
        data = file->map(0, size);
        if (data == NULL)
            qFatal("Unable to map gallery: %s", qPrintable(url));
    }
};

static QSharedPointer<Distance> distance;
static QList<MappedGallery> galleries;

static void compare_utemplates(br_const_utemplate target, br_callback_context context)
{
    TopTargets *topTargets = (TopTargets*) context;
    topTargets->tryAdd(target, distance->compare(target->data, topTargets->query->data, 768));
}

static void search_utemplate(br_const_utemplate query, br_callback_context)
{
    TopTargets *topTargets = new TopTargets(query);
    foreach (const MappedGallery &gallery, galleries)
        br_iterate_utemplates(reinterpret_cast<br_const_utemplate>(gallery.data), reinterpret_cast<br_const_utemplate>(gallery.data + gallery.size), compare_utemplates, topTargets);
    topTargets->print();
    delete topTargets;
}

int main(int argc, char *argv[])
{
    QStringList urls;
    for (int i=1; i<argc; i++) {
        if      (!strcmp(argv[i], "-help" ))     { help(); exit(EXIT_SUCCESS); }
        else if (!strcmp(argv[i], "-limit"))     limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-threshold")) threshold = atof(argv[++i]);
        else                                     urls.append(argv[i]);
    }

    Context::initialize(argc, argv, "", false);

    foreach (const QString &url, urls)
        galleries.append(MappedGallery(url));

    Globals->quiet = true;
    distance = Distance::fromAlgorithm("FaceRecognition");
    br_iterate_utemplates_file(stdin, search_utemplate, NULL);

    Context::finalize();
    return EXIT_SUCCESS;
}
