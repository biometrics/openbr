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
#include <iomanip>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>
#include <openbr/openbr_plugin.h>
#include <openbr/universal_template.h>

using namespace br;
using namespace cv;
using namespace std;

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
           "For every template read from _stdin_, search writes the top sorted matches a newline-terminated JSON object to _stdout_.\n"
           "The JSON object will include at least `AlgorithmID`, (query) `ImageID`, (query) `TemplateID`, `Targets` and any algorithm-specific metadata fields set during _enroll_.\n"
           "\n"
           "Optional Arguments\n"
           "------------------\n"
           "* -help              - Print usage information.\n"
           "* -limit <int>       - Maximum number of returns (20 otherwise).\n"
           "* -threshold <float> - Minimum similarity score (none otherwise).");
}

static size_t limit = 20;
static float threshold = -numeric_limits<float>::max();

struct SearchResults
{
    typedef pair<float, br_const_utemplate> Target;
    vector<Target> topTargets;
    br_const_utemplate query;

    SearchResults(br_const_utemplate query)
        : query(query) {}

    virtual ~SearchResults() {}

    void consider(br_const_utemplate target)
    {
        const float score = compare(target, query);
        if ((score < threshold) || ((topTargets.size() == limit) && (score < topTargets.front().first)))
            return;

        topTargets.push_back(Target(score, target));
        make_heap(topTargets.begin(), topTargets.end());

        if (topTargets.size() == limit + 1)
            pop_heap(topTargets.begin(), topTargets.end());
    }

    static void writeMD5asHex(const unsigned char *md5)
    {
        cout << hex << setfill('0');
        for (int i=0; i<16; i++)
            cout << setw(2) << md5[i];
        cout << dec;
    }

    void print()
    {
        sort_heap(topTargets.begin(), topTargets.end());

        cout << "{ \"AlgorithmID\"=" << query->algorithmID;
        cout << ", \"QueryImageID\"=";
        writeMD5asHex(query->imageID);
        cout << ", \"QueryTemplateID\"=";
        writeMD5asHex(query->templateID);
        printMetadata(query);
        cout << ", \"Targets\"=[ ";
        for (int i=topTargets.size()-1; i>=0; i--) {
            Target &target = topTargets[i];
            cout  << "{ \"ImageID\"=";
            writeMD5asHex(target.second->imageID);
            cout << ", \"TemplateID\"=";
            writeMD5asHex(target.second->templateID);
            cout << ", \"Score\"=" << target.first;
            printMetadata(target.second);
            cout << " }";
            if (i > 0)
                cout << ", ";
        }
        cout << "]}\n" << flush;
    }

    virtual float compare(br_const_utemplate target, br_const_utemplate query) const = 0;
    virtual void printMetadata(br_const_utemplate) const { return; }
};

struct FaceRecognition : public SearchResults
{
    QSharedPointer<Distance> algorithm;

    FaceRecognition(br_const_utemplate query)
        : SearchResults(query)
    {
        algorithm = Distance::fromAlgorithm("FaceRecognition");
    }

    float compare(br_const_utemplate target, br_const_utemplate query) const
    {
        return algorithm->compare(target->data, query->data, 768);
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

static QList<MappedGallery> galleries;

static void compare_utemplates(br_const_utemplate target, br_callback_context context)
{
    SearchResults *searchResults = (SearchResults*) context;
    searchResults->consider(target);
}

static void search_utemplate(br_const_utemplate query, br_callback_context)
{
    SearchResults *searchResults = new FaceRecognition(query);
    foreach (const MappedGallery &gallery, galleries)
        br_iterate_utemplates(reinterpret_cast<br_const_utemplate>(gallery.data), reinterpret_cast<br_const_utemplate>(gallery.data + gallery.size), compare_utemplates, searchResults);
    searchResults->print();
    delete searchResults;
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
    br_iterate_utemplates_file(stdin, search_utemplate, NULL);

    Context::finalize();
    return EXIT_SUCCESS;
}
