#include <QBrush>
#include <QColor>
#include <QFileInfo>
#include <QFileDialog>
#include <QIcon>
#include <QPen>
#include <openbr.h>

#include "tail.h"

using namespace br;

/**** TAIL ****/
/*** PUBLIC ***/
Tail::Tail(QWidget *parent)
    : QMainWindow(parent)
{
    tbFirst.setIcon(QIcon(":/glyphicons/png/glyphicons_171_fast_backward@2x.png"));
    tbFirst.setToolTip("First");
    tbPrevious.setIcon(QIcon(":/glyphicons/png/glyphicons_170_step_backward@2x.png"));
    tbPrevious.setToolTip("Previous");
    tbNext.setIcon(QIcon(":/glyphicons/png/glyphicons_178_step_forward@2x.png"));
    tbNext.setToolTip("Next");
    tbLast.setIcon(QIcon(":/glyphicons/png/glyphicons_177_fast_forward@2x.png"));
    tbLast.setToolTip("Last");
    wLeftSpacer.setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    wRightSpacer.setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    tbToolBar.addWidget(&dDataset);
    tbToolBar.addWidget(&wLeftSpacer);
    tbToolBar.addWidget(&tbFirst);
    tbToolBar.addWidget(&tbPrevious);
    tbToolBar.addWidget(&tbNext);
    tbToolBar.addWidget(&tbLast);
    tbToolBar.addWidget(&wRightSpacer);
    tbToolBar.addWidget(&sScore);
    tbToolBar.setIconSize(QSize(20,20));

    addToolBar(Qt::BottomToolBarArea, &tbToolBar);
    setCentralWidget(&sdScoreDistribution);
    setWindowFlags(Qt::Widget);

    count = 9;
    index = 0;
    targetLocked = queryLocked = false;
    updateInterface();

    connect(&dDataset, SIGNAL(newDistribution(QString)), &sdScoreDistribution, SLOT(setDistribution(QString)));
    connect(&sdScoreDistribution, SIGNAL(newThreshold(float)), this, SLOT(setThreshold(float)));
    connect(&tbFirst, SIGNAL(clicked()), this, SLOT(first()));
    connect(&tbPrevious, SIGNAL(clicked()), this, SLOT(previous()));
    connect(&tbNext, SIGNAL(clicked()), this, SLOT(next()));
    connect(&tbLast, SIGNAL(clicked()), this, SLOT(last()));
}

/*** PUBLIC SLOTS ***/
void Tail::setAlgorithm(const QString &algorithm)
{
    dDataset.setAlgorithm(algorithm);
    compare();
}

void Tail::setIndex(int index)
{
    if (index < 0) index = 0;
    if (index > scores.size()-1) index = std::max(0, scores.size()-1);
    if (index > scores.size() - count) index = std::max(0, scores.size() - count);
    this->index = index;
    updateInterface();

    emit newTargetFiles(targets.mid(index, count));
    emit newQueryFiles(queries.mid(index, count));
}

void Tail::setTargetGallery(const File &gallery)
{
    target = gallery;
    compare();
}

void Tail::setQueryGallery(const File &gallery)
{
    query = gallery;
    compare();
}

void Tail::setTargetGalleryFiles(const br::FileList &files)
{
    targets = files;
    setIndex(index);
}

void Tail::setQueryGalleryFiles(const br::FileList &files)
{
    queries = files;
    setIndex(index);
}

void Tail::setCount(int count)
{
    this->count = count;
    setIndex(index);
}

void Tail::setThreshold(float score)
{
    int nearestIndex = -1;
    float nearestDistance = std::numeric_limits<float>::max();
    for (int i=0; i<scores.size(); i++) {
        float distance = std::abs(scores[i] - score);
        if (distance < nearestDistance) {
            nearestIndex = i;
            nearestDistance = distance;
        }
    }
    setIndex(nearestIndex);
}

/*** PROTECTED ***/
void Tail::keyPressEvent(QKeyEvent *event)
{
    QWidget::keyPressEvent(event);
    event->accept();
    if      ((event->key() == Qt::Key_Up)    || (event->key() == Qt::Key_W)) first();
    else if ((event->key() == Qt::Key_Left)  || (event->key() == Qt::Key_A)) previous();
    else if ((event->key() == Qt::Key_Right) || (event->key() == Qt::Key_D)) next();
    else if ((event->key() == Qt::Key_Down)  || (event->key() == Qt::Key_S)) last();
}

void Tail::wheelEvent(QWheelEvent *event)
{
    QWidget::wheelEvent(event);
    event->accept();
    if (event->delta() < 0) next();
    else                    previous();
}

/*** PRIVATE ***/
void Tail::compare()
{
    if (target.isNull() || query.isNull()) return;
    QString tail = QString("%1/comparisons/%2_%3.tail[atMost=5000,threshold=1,args,Cache]").arg(br_scratch_path(), qPrintable(target.baseName()+target.hash()), qPrintable(query.baseName()+query.hash()));
    Compare(target.flat(), query.flat(), tail);
    import(tail);
    QFile::remove(tail);
}

/*** PRIVATE SLOTS ***/
void Tail::updateInterface()
{
    tbFirst.setEnabled(index < scores.size() - count);
    tbPrevious.setEnabled(index < scores.size() - count);
    tbNext.setEnabled(index > 0);
    tbLast.setEnabled(index > 0);

    if (scores.isEmpty()) {
        sScore.clear();
    } else {
        sScore.setScore(scores[index]);
        sdScoreDistribution.setThreshold(scores[index]);
    }
}

void Tail::import(QString tailFile)
{
    if (tailFile.isEmpty()) {
        tailFile = QFileDialog::getOpenFileName(this, "Import Tail File");
        if (tailFile.isEmpty()) return;
    }
    tailFile = tailFile.left(tailFile.indexOf('['));

    QFile file(tailFile);
    file.open(QFile::ReadOnly);
    QStringList lines = QString(file.readAll()).split('\n');
    lines.takeFirst(); // Remove header
    file.close();

    targets.clear();
    queries.clear();
    scores.clear();
    foreach (const QString &line, lines) {
        QStringList words = Object::parse(line);
        if (words.size() != 3) continue;
        bool ok;
        float score = words[0].toFloat(&ok); assert(ok);
        targets.append(words[1]);
        queries.append(words[2]);
        scores.append(score);
    }

    sdScoreDistribution.setLiveScores(scores);
    setIndex(0);
}

void Tail::first()
{
    setIndex(scores.size()-1);
}

void Tail::previous()
{
    setIndex(index + count);
}

void Tail::next()
{
    setIndex(index - count);
}

void Tail::last()
{
    setIndex(0);
}

void Tail::lock(bool checked)
{
    tbFirst.setEnabled(!checked);
    tbLast.setEnabled(!checked);
}

void Tail::selected(QPointF point)
{
    int nearestIndex = -1;
    float nearestDistance = std::numeric_limits<float>::max();
    for (int i=0; i<scores.size(); i++) {
        float distance = std::abs(scores[i] - point.x());
        if (distance < nearestDistance) {
            nearestIndex = i;
            nearestDistance = distance;
        }
    }
    setIndex(nearestIndex);
}

#include "moc_tail.cpp"
