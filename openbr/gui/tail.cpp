#include <QtConcurrentRun>

#include "tail.h"
#include <assert.h>

using namespace br;

/**** TAIL ****/
/*** PUBLIC ***/
Tail::Tail(QWidget *parent)
    : QWidget(parent)
{
    layout = new QHBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    slider = new QSlider(this);
    slider->setOrientation(Qt::Horizontal);
    lhs = new QLabel(this);
    rhs = new QLabel(this);
    layout->addWidget(lhs);
    layout->addWidget(slider, 1);
    layout->addWidget(rhs);
    setFocusPolicy(Qt::StrongFocus);
    setVisible(false);
    connect(slider, SIGNAL(valueChanged(int)), this, SLOT(setIndex(int)));
    connect(&compareWatcher, SIGNAL(finished()), this, SLOT(compareDone()));
}

/*** PUBLIC SLOTS ***/
void Tail::clear()
{
    targetGallery = File();
    queryGallery = File();
    targetFiles.clear();
    queryFiles.clear();
    scores.clear();
    slider->setMaximum(0);
    setIndex(0);
    setVisible(false);
}

void Tail::setIndex(int index)
{
    index = std::min(std::max(slider->minimum(), index), slider->maximum());
    slider->setValue(index);
    emit newTargetFile((index >= 0) && (index < targetFiles.size()) ? targetFiles[index] : File());
    emit newQueryFile((index >= 0) && (index < queryFiles.size()) ? queryFiles[index] : File());
    emit newScore((index >= 0) && (index < scores.size()) ? scores[index] : std::numeric_limits<float>::quiet_NaN());
}

void Tail::setTargetGallery(const File &gallery)
{
    targetGallery = gallery;
    compare();
}

void Tail::setQueryGallery(const File &gallery)
{
    queryGallery = gallery;
    compare();
}

/*** PROTECTED ***/
void Tail::keyPressEvent(QKeyEvent *event)
{
    QWidget::keyPressEvent(event);
    event->accept();
    if      (event->key() == Qt::Key_Up)    first();
    else if (event->key() == Qt::Key_Left)  previous();
    else if (event->key() == Qt::Key_Right) next();
    else if (event->key() == Qt::Key_Down)  last();
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
    targetFiles.clear();
    queryFiles.clear();
    scores.clear();

    if (targetGallery.isNull() || queryGallery.isNull()) {
        if (!targetGallery.isNull()) targetFiles = TemplateList::fromGallery(targetGallery).files();
        if (!queryGallery.isNull()) queryFiles = TemplateList::fromGallery(queryGallery).files();
        slider->setMaximum(std::max(targetFiles.size(), queryFiles.size()) - 1);
        lhs->setText("First Image");
        rhs->setText("Last Image");
        setVisible(slider->maximum() > 1);
        setIndex(0);
    } else {
        compareWatcher.setFuture(QtConcurrent::run(Compare, targetGallery.flat(), queryGallery.flat(), QString("buffer.tail[atMost=1000]")));
    }
}

void Tail::first()
{
    setIndex(0);
}

void Tail::previous()
{
    setIndex(slider->value()-1);
}

void Tail::next()
{
    setIndex(slider->value()+1);
}

void Tail::last()
{
    setIndex(scores.size()-1);
}

void Tail::compareDone()
{
    QStringList lines = QString(Globals->buffer).split('\n');
    lines.takeFirst(); // Remove header

    foreach (const QString &line, lines) {
        const QStringList words = Object::parse(line);
        if (words.size() != 3) qFatal("Invalid tail file.");
        bool ok;
        float score = words[0].toFloat(&ok); assert(ok);
        targetFiles.append(words[1]);
        queryFiles.append(words[2]);
        scores.append(score);
    }
    slider->setMaximum(scores.size()-1);
    lhs->setText("Best Match");
    rhs->setText("Worst Match");

    setVisible(slider->maximum() > 1);
    setIndex(0);
}

#include "moc_tail.cpp"
