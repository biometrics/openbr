#include "metadata.h"

using namespace br;

Metadata::Metadata(QWidget *parent) :
    QWidget(parent)
{
    name = new QLabel("Name: ");
    gender = new QLabel("Gender: ");
    age = new QLabel("Age: ");
    race = new QLabel("Race: ");
    height = new QLabel("Height: ");
    weight = new QLabel("Weight: ");

    layout.addWidget(name);
    layout.addWidget(gender);
    layout.addWidget(age);
    layout.addWidget(race);
    layout.addWidget(height);
    layout.addWidget(weight);

    setLayout(&layout);
}

void Metadata::reset()
{
    name->setText("Name: ");
    gender->setText("Gender: ");
    age->setText("Age: ");
    race->setText("Race: ");
    height->setText("Height: ");
    weight->setText("Weight: ");
}

void Metadata::setMetadata(br::File file)
{
    if (file.isNull()) {
        reset();
    }
    else {
        name->setText(file.get<QString>("LASTNAME", "N/A") + ", " + file.get<QString>("FIRSTNAME", "N/A"));
        gender->setText("Gender: " + file.get<QString>("GENDER", "N/A"));
        age->setText("Age: " + file.get<QString>("Age", "N/A"));
        race->setText("Race: " + file.get<QString>("RACE", "N/A"));
        height->setText("Height: " + file.get<QString>("HEIGHT", "N/A"));
        weight->setText("Weight: " + file.get<QString>("WEIGHT", "N/A"));
    }
}
