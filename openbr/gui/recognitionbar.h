#ifndef BR_RECOGNITIONBAR_H
#define BR_RECOGNITIONBAR_H

#include <QToolBar>
#include <QPushButton>
#include <QComboBox>
#include <QActionGroup>
#include <QAction>
#include <QLabel>
#include <QToolButton>
#include <QCheckBox>

#include <openbr/gui/formcombowidget.h>
#include <openbr/gui/rangewidget.h>
#include <openbr/gui/pageflipwidget.h>
#include <openbr/gui/searchboxwidget.h>
#include <openbr/gui/metadata.h>

#include <openbr/openbr_plugin.h>

namespace br
{

class BR_EXPORT RecognitionBar : public QToolBar
{
    Q_OBJECT

    QCheckBox *landmarkCheck;

    QCheckBox *maleCheck;
    QCheckBox *femaleCheck;

    QCheckBox *whiteCheck;
    QCheckBox *blackCheck;
    QCheckBox *hispanicCheck;
    QCheckBox *asianCheck;
    QCheckBox *otherCheck;

    QCheckBox *ageCheck;

    QLabel *demographicsLabel;
    QLabel *algorithmLabel;
    QLabel *findLabel;

    FormComboWidget *model;

    RangeWidget *rangeWidget;
    SearchBoxWidget *searchBox;
    QPushButton *compareButton;

    QWidget *spacer;

public:
    explicit RecognitionBar(QWidget *parent = 0);
    
    Metadata *metadata;

public slots:
    void addAlgorithm();

private slots:
    void genderChanged();
    void raceChanged();
    void ageRangeChanged();

signals:
    void newAlgorithm(QString);
    void newDemographics(QString, QStringList);
    void newIndex(int);
    void setFiles(br::FileList);
    void compare();

    void showLandmarks(bool);
};

} // namespace br

#endif // BR_RECOGNITIONBAR_H
