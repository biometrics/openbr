#ifndef BR_METADATA_H
#define BR_METADATA_H

#include <QToolBar>
#include <QHBoxLayout>
#include <QLabel>
#include <QGroupBox>

#include <openbr/openbr_plugin.h>

namespace br {

class BR_EXPORT Metadata : public QWidget
{
    Q_OBJECT

    QVBoxLayout layout;

    QLabel *name;
    QLabel *gender;
    QLabel *race;
    QLabel *age;
    QLabel *weight;
    QLabel *height;

public:
    explicit Metadata(QWidget *parent = 0);
    
signals:
    
public slots:
    void reset();
    void setMetadata(br::File);
    
};

}

#endif // BR_METADATA_H
