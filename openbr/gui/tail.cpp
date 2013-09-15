#include <QFileDialog>

#include "tail.h"

using namespace br;

/**** TAIL ****/
/*** PUBLIC ***/
Tail::Tail(QWidget *parent)
    : QSlider(parent)
{
    count = 1;
    setOrientation(Qt::Horizontal);
    setVisible(false);
}

/*** PUBLIC SLOTS ***/
void Tail::setIndex(int index)
{
    if (index < 0) index = 0;
    if (index >= scores.size()) index = std::max(0, scores.size()-1);
    if (index > scores.size() - count) index = std::max(0, scores.size() - count);
    setIndex(index);

    emit newTargetFile(targets[index]);
    emit newTargetFiles(targets.mid(index, count));
    emit newQueryFile(queries[index]);
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

void Tail::setTargetGalleryFiles(const FileList &files)
{
    targets = files;
    setIndex(value());
}

void Tail::setQueryGalleryFiles(const FileList &files)
{
    queries = files;
    setIndex(value());
}

void Tail::setCount(int count)
{
    this->count = count;
    setIndex(value());
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
    QString tail = QString("%1/comparisons/%2_%3.tail[atMost=5000,threshold=1,args,cache]").arg(br_scratch_path(), qPrintable(target.baseName()+target.hash()), qPrintable(query.baseName()+query.hash()));
    Compare(target.flat(), query.flat(), tail);
    import(tail);
    QFile::remove(tail);
}

/*** PRIVATE SLOTS ***/
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

    setMaximum(scores.size()-1);
    setVisible(scores.size() > 0);
    setIndex(0);
}

void Tail::first()
{
    setIndex(scores.size()-1);
}

void Tail::previous()
{
    setIndex(value() + count);
}

void Tail::next()
{
    setIndex(value() - count);
}

void Tail::last()
{
    setIndex(0);
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
