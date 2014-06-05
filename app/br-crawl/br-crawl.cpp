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

static void help()
{
    printf("br-crawl [URL] [args]\n"
           "=====================\n"
           "* __stdin__  - URLs\n"
           "* __stdout__ - Image URLs/JSON\n"
           "\n"
           "_br-crawl_ conducts a recursive descent search for images from a root URL.\n"
           "Crawl will read root URLs from _stdin_ if none are provided.\n"
           "Crawl writes every discovered image URL in a new line to _stdout_.\n"
           "Arguments specifiying the duration of crawl are on a per-root-URL basis.\n"
           "\n"
           "Crawl identifies image URLs based on known image file extensions like `.png`.\n"
           "Crawl is not expected to verify that URLs are images and may produce false positives.\n"
           "\n"
           "Optional Arguments\n"
           "------------------\n"
           "* -auto         - Crawl chooses its own root URL (must be specified otherwise).\n"
           "* -depth <int>  - The levels to recursively search (unlimited otherwise).\n"
           "* -depthFirst   - Depth-first search (breadth-first otherwise).\n"
           "* -help         - Print usage information.\n"
           "* -images <int> - The number of image URLs to obtain (unlimited otherwise).\n"
           "* -json         - Output JSON instead or URLs.\n"
           "* -time <int>   - The seconds to spend searching for images (unlimited otherwise).\n");
}

static const char *root = NULL;
static bool autoRoot = false;
static int depth = INT_MAX;
static bool depthFirst = false;
static int images = INT_MAX;
static bool json = false;
static int timeLimit = INT_MAX;

static QTime elapsed;
static int currentImages = 0;

static void crawl(QFileInfo url, int currentDepth = 0)
{
    if ((currentImages >= images) || (currentDepth >= depth) || (elapsed.elapsed()/1000 >= timeLimit))
        return;

    if (url.filePath().startsWith("file://"))
        url = QFileInfo(url.filePath().mid(7));

    if (url.isDir()) {
        const QDir dir(url.absoluteFilePath());
        const QFileInfoList files = dir.entryInfoList(QDir::Files);
        const QFileInfoList subdirs = dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot);
        foreach (const QFileInfo &first, depthFirst ? subdirs : files)
            crawl(first, currentDepth + 1);
        foreach (const QFileInfo &second, depthFirst ? files : subdirs)
            crawl(second, currentDepth + 1);
    } else if (url.isFile()) {
        const QString suffix = url.suffix();
        if ((suffix == "bmp") || (suffix == "jpg") || (suffix == "jpeg") || (suffix == "png") || (suffix == "tiff")) {
            printf(json ? "{ URL = \"file://%s\" }\n" : "file://%s\n", qPrintable(url.canonicalFilePath()));
            fflush(stdout);
            currentImages++;
        }
    }
}

int main(int argc, char *argv[])
{
    for (int i=1; i<argc; i++) {
        if      (!strcmp(argv[i], "-auto"      )) autoRoot = true;
        else if (!strcmp(argv[i], "-depth"     )) depth = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-depthFirst")) depthFirst = true;
        else if (!strcmp(argv[i], "-help"      )) { help(); exit(EXIT_SUCCESS); }
        else if (!strcmp(argv[i], "-images"    )) images = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-json"      )) json = true;
        else if (!strcmp(argv[i], "-time"      )) timeLimit = atoi(argv[++i]);
        else                                      root = argv[i];
    }

    elapsed.start();
    if (root != NULL) {
        crawl(QFileInfo(root));
    } else {
        if (autoRoot) {
            foreach (const QString &path, QStandardPaths::standardLocations(QStandardPaths::HomeLocation))
                crawl(path);
        } else {
            QFile file;
            file.open(stdin, QFile::ReadOnly);
            while (!file.atEnd()) {
                const QString url = QString::fromLocal8Bit(file.readLine()).simplified();
                if (!url.isEmpty())
                    crawl(url);
            }
        }
    }

    return EXIT_SUCCESS;
}
