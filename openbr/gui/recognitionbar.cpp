#include "recognitionbar.h"

#include <QFileDialog>

using namespace br;

RecognitionBar::RecognitionBar(QWidget *parent) :
    QToolBar(parent)
{
    setToolButtonStyle(Qt::ToolButtonTextOnly);

    addSeparator();

    algorithmLabel = new QLabel(this);
    algorithmLabel->setAlignment(Qt::AlignCenter);
    algorithmLabel->setTextFormat(Qt::RichText);
    algorithmLabel->setText("<img src=:/algorithm.png> Algorithm");
    addWidget(algorithmLabel);

    model = new FormComboWidget("Model: ", this);
    addWidget(model);

    connect(model, SIGNAL(currentIndexChanged(QString)), this, SIGNAL(newAlgorithm(QString)));

    landmarkCheck = new QCheckBox("Show Landmarks", this);
    addWidget(landmarkCheck);

    connect(landmarkCheck, SIGNAL(clicked(bool)), this, SIGNAL(showLandmarks(bool)));

    addSeparator();

    demographicsLabel = new QLabel(this);
    demographicsLabel->setAlignment(Qt::AlignCenter);
    demographicsLabel->setTextFormat(Qt::RichText);
    demographicsLabel->setText("<img src=:/demographics.png> Demographics");
    addWidget(demographicsLabel);

    maleCheck = new QCheckBox("Male", this);
    addWidget(maleCheck);
    femaleCheck = new QCheckBox("Female", this);
    addWidget(femaleCheck);

    connect(maleCheck, SIGNAL(clicked()), this, SLOT(genderChanged()));
    connect(femaleCheck, SIGNAL(clicked()), this, SLOT(genderChanged()));

    addSeparator();

    whiteCheck = new QCheckBox("White", this);
    addWidget(whiteCheck);
    connect(whiteCheck, SIGNAL(clicked()), this, SLOT(raceChanged()));
    blackCheck = new QCheckBox("Black", this);
    addWidget(blackCheck);
    connect(blackCheck, SIGNAL(clicked()), this, SLOT(raceChanged()));
    hispanicCheck = new QCheckBox("Hispanic", this);
    addWidget(hispanicCheck);
    connect(hispanicCheck, SIGNAL(clicked()), this, SLOT(raceChanged()));
    asianCheck = new QCheckBox("Oriental/Asian", this);
    addWidget(asianCheck);
    connect(hispanicCheck, SIGNAL(clicked()), this, SLOT(raceChanged()));
    otherCheck = new QCheckBox("Other", this);
    addWidget(otherCheck);
    connect(otherCheck, SIGNAL(clicked()), this, SLOT(raceChanged()));

    addSeparator();

    ageCheck = new QCheckBox("Age", this);
    addWidget(ageCheck);
    connect(ageCheck, SIGNAL(clicked()), this, SLOT(ageRangeChanged()));

    rangeWidget = new RangeWidget(this);
    connect(rangeWidget, SIGNAL(newRange(int,int)), this, SLOT(ageRangeChanged()));
    addWidget(rangeWidget);

    addSeparator();

    findLabel = new QLabel(this);
    findLabel->setAlignment(Qt::AlignCenter);
    findLabel->setTextFormat(Qt::RichText);
    findLabel->setText("<img src=:/sketch.png> Find");

    addWidget(findLabel);

    searchBox = new SearchBoxWidget(this);
    connect(this, SIGNAL(setFiles(br::FileList)), searchBox, SLOT(setFiles(br::FileList)));
    connect(searchBox, SIGNAL(newIndex(int)), this, SIGNAL(newIndex(int)));

    addWidget(searchBox);

    metadata = new Metadata(this);

    addWidget(metadata);

    spacer = new QWidget(this);
    spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    addWidget(spacer);

    compareButton = new QPushButton(this);
    compareButton->setText("Compare");
    compareButton->setIcon(QIcon(":/compare.png"));
    addWidget(compareButton);
    compareButton->setFocus();

    connect(compareButton, SIGNAL(clicked()), this, SIGNAL(compare()));

    setOrientation(Qt::Vertical);

    addSeparator();

    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Minimum);
}

void RecognitionBar::addAlgorithm()
{
    model->addItem(QFileDialog::getOpenFileName(this, "Add a model...", Context::scratchPath()));
}

void RecognitionBar::genderChanged()
{
    QStringList genders;

    if (maleCheck->isChecked()) genders.push_back(maleCheck->text()[0].toUpper());
    if (femaleCheck->isChecked()) genders.push_back(femaleCheck->text()[0].toUpper());

    emit newDemographics("GENDER", genders);
}

void RecognitionBar::raceChanged()
{
    QStringList races;

    if (whiteCheck->isChecked()) races.push_back(whiteCheck->text().toUpper());
    if (blackCheck->isChecked()) races.push_back(blackCheck->text().toUpper());
    if (hispanicCheck->isChecked()) races.push_back(hispanicCheck->text().toUpper());
    if (asianCheck->isChecked()) races.push_back(asianCheck->text().toUpper());
    if (otherCheck->isChecked()) races.push_back(otherCheck->text().toUpper());

    emit newDemographics("RACE", races);
}

void RecognitionBar::ageRangeChanged()
{
    QStringList range;

    if (ageCheck->isChecked()) {
        int age = rangeWidget->getLowerBound();
        while (age <= rangeWidget->getUpperBound()) {
            range.push_back(QString::number(age));
            age++;
        }
    }
    else range.clear();

    emit newDemographics("Age", range);
}

#include "moc_recognitionbar.cpp"
