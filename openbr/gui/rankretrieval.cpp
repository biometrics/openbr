#include "rankretrieval.h"

#include <QtConcurrent>

#include <openbr/openbr.h>
#include <openbr/gui/faceviewer.h>

using namespace br;

RankRetrieval::RankRetrieval(QWidget *parent) :
    QWidget(parent),
    gridPage(0),
    gridSize(9)
{
    targetPath = Context::scratchPath();

    connect(&compareWatcher, SIGNAL(finished()), this, SLOT(compareDone()));
}

void RankRetrieval::clear()
{

}

void RankRetrieval::setAlgorithm(const QString &algorithm)
{
    br_set_property("algorithm",qPrintable(algorithm));
}

void RankRetrieval::setTargetGallery(const File &file)
{
    target = file;
    enroll();
}

void RankRetrieval::setTargetPath()
{
    targetPath = QFileDialog::getSaveFileName(this, "Specify Target Gallery...", targetPath, tr(".gal Files (*.gal)"));
}

void RankRetrieval::setQueryGallery(const File &file)
{
    QSharedPointer<Transform> transform = Transform::fromAlgorithm("algorithm");
    Template queryTemplate(file);
    queryTemplate >> *transform;

    query = queryTemplate;

    emit newQueryFile(query);
}

void RankRetrieval::enroll()
{
    File targetGallery(targetPath + ".gal");
    targetGallery.set("append", true);

    enrollWatcher.setFuture(QtConcurrent::run(Enroll, target.flat(), targetGallery.flat()));
}

void RankRetrieval::compare()
{
    File targetGallery(targetPath + ".gal");

    compareWatcher.setFuture(QtConcurrent::run(Compare, targetGallery.flat(), query.flat(), QString("buffer.rr[byLine,limit=200]")));
}

void RankRetrieval::first()
{
    if (matches.isEmpty() || gridPage == 0) return;

    gridPage = 0;

    display();
}

void RankRetrieval::previous()
{
    if (matches.isEmpty() || gridPage == 0) return;

    gridPage--;

    display();
}

void RankRetrieval::next()
{
    if (matches.isEmpty() || matches.size() <= gridPage*gridSize+gridSize) return;

    gridPage++;

    display();
}

void RankRetrieval::last()
{
    if (matches.isEmpty() || gridPage == matches.size()/gridSize) return;

    gridPage = ceil((float)matches.size()/(float)gridSize)-1;

    display();
}

void RankRetrieval::setGridSize(const QString &size)
{
    gridSize = size.toInt();
    gridPage = 0;

    display();
}

void RankRetrieval::setIndex(int index)
{
    (void) index;
}

void RankRetrieval::compareDone()
{    
    matches.clear();

    QStringList results = QString(Globals->buffer).split("\n");

    if (results.empty()) {
        qWarning("Error: No successful matches.");
        return;
    }

    foreach (const QString &line, results) {
        File match(line);
        matches.append(match);
    }

    display();
}

void RankRetrieval::display()
{
    FileList display;

    for(int i = gridPage*gridSize; i < gridPage*gridSize+gridSize; i++) {
        if (i >= matches.size()) break;
        display.append(matches.at(i));
    }

    emit newTargetFileList(display);
}

#include "moc_rankretrieval.cpp"
