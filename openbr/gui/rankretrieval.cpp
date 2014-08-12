#include "rankretrieval.h"

#include <QtConcurrent>

#include <openbr/openbr.h>
#include <openbr/gui/faceviewer.h>

using namespace br;

RankRetrieval::RankRetrieval(QWidget *parent) :
    QWidget(parent)
{
    targetPath = Context::scratchPath();
}

void RankRetrieval::clear()
{

}

// This should be changed to used algorithm.cpp
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
    query = file;

    // Change this to the set algorithm
    QSharedPointer<Transform> transform = br::Transform::fromAlgorithm("FaceDetection+FaceRecognitionRegistration");
    Template queryTemplate(query);
    queryTemplate >> *transform;

    emit newQueryFile(queryTemplate.file);
}

void RankRetrieval::enroll()
{
    File targetGallery(targetPath + ".gal");
    targetGallery.set("append", true);

    enrollWatcher.setFuture(QtConcurrent::run(Enroll, targetGallery.flat(), target.flat()));
}

void RankRetrieval::compare()
{

}

void RankRetrieval::first()
{
    /*if (matches.isEmpty() || gridPage == 0 || comparing) return;

    gridPage = 0;

    emit clearSelection();

    displayMugshots();*/
}

void RankRetrieval::previous()
{
    /*if (matches.isEmpty() || gridPage == 0 || comparing) return;

    gridPage--;

    emit clearSelection();

    displayMugshots();*/
}

void RankRetrieval::next()
{
    /*if (matches.isEmpty() || matches.size() <= gridPage*gridSize+gridSize || comparing) return;

    gridPage++;

    emit clearSelection();

    displayMugshots();*/
}

void RankRetrieval::last()
{
    /*if (matches.isEmpty() || gridPage == matches.size()/gridSize || comparing) return;

    gridPage = matches.size()/gridSize;

    emit clearSelection();

    displayMugshots();*/
}

void RankRetrieval::setIndex(int index)
{

}

void RankRetrieval::compareDone()
{
    /*
    if (matches.isEmpty()) return;

    br::FileList displayMugshots;
    QStringList labels;

    for(int i = gridPage*gridSize; i < gridPage*gridSize+gridSize; i++) {
        if (i >= matches.size()) {
            break;
        }

        displayMugshots.push_back(matches[i]);
        labels.push_back("Rank: " + QString::number(i+1) + "\nScore: " + QString::number(scores[i], 'f', 3));
    }

    if (index != -1) heatMap(index);

    emit newMugshotLineup(displayMugshots, index);
    emit newFormat(targetFormat);
    emit newLabels(labels);*/
}

#include "moc_rankretrieval.cpp"
