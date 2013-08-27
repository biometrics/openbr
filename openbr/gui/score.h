#ifndef BR_SCORE_H
#define BR_SCORE_H

#include <QLabel>
#include <QWidget>
#include <openbr/openbr_export.h>

namespace br
{

class BR_EXPORT Score : public QLabel
{
    Q_OBJECT

public:
    explicit Score(QWidget *parent = 0);

public slots:
    void setScore(float score);
};

}

#endif // BR_SCORE_H
