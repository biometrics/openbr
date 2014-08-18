#ifndef BR_PAGEFLIPWIDGET_H
#define BR_PAGEFLIPWIDGET_H

#include <QWidget>
#include <QPushButton>
#include <QHBoxLayout>

#include <openbr/openbr_export.h>

namespace br {

class BR_EXPORT PageFlipWidget : public QWidget
{
    Q_OBJECT

    QHBoxLayout boxLayout;

    QPushButton *firstPage;
    QPushButton *previousPage;
    QPushButton *nextPage;
    QPushButton *lastPage;

public:
    explicit PageFlipWidget(QWidget *parent = 0);

signals:

    void first();
    void next();
    void previous();
    void last();

public slots:
    
};

} // namespace br

#endif // BR_PAGEFLIPWIDGET_H
