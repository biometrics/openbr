#include <QDebug>
#include "score.h"

/**** SCORE ****/
/*** PUBLIC ***/
br::Score::Score(QWidget *parent)
    : QLabel(parent)
{
    setToolTip("Similarity Score");
}

/*** PUBLIC SLOTS ***/
void br::Score::setScore(float score)
{
    setText("<b>Similarity:</b> " + QString::number(score, 'f', 2));
}

#include "moc_score.cpp"
