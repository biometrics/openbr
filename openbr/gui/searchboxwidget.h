#ifndef BR_SEARCHBOXWIDGET_H
#define BR_SEARCHBOXWIDGET_H

#include <QWidget>
#include <QLineEdit>
#include <QCompleter>
#include <QVBoxLayout>
#include <QStringListModel>

#include <openbr/openbr_plugin.h>

namespace br {

class BR_EXPORT SearchBoxWidget : public QWidget
{
    Q_OBJECT

    QLineEdit *searchBar;
    QCompleter *completer;
    QStringListModel *model;
    QVBoxLayout *layout;

    QStringList words;
    int startIndex;

public:
    explicit SearchBoxWidget(QWidget *parent = 0);
    
signals:
    
    void newIndex(int);

public slots:
    
    void setFiles(br::FileList files);
    void setIndex();
};

}

#endif // BR_SEARCHBOXWIDGET_H
