#include <QtConcurrentRun>
#include <openbr/openbr_plugin.h>

#include "classifier.h"

using namespace br;

/**** CLASSIFIER ****/
/*** PUBLIC ***/
Classifier::Classifier(QWidget *parent)
    : QLabel(parent)
{
    setAlignment(Qt::AlignCenter);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    connect(this, SIGNAL(newClassification(QString,QString)), this, SLOT(setClassification(QString,QString)));
}

void Classifier::setAlgorithm(const QString &algorithm)
{
    this->algorithm = algorithm;
    _classify(File()); // Trigger algorithm initialization
}

/*** PUBLIC SLOTS ***/
void Classifier::classify(const File &file)
{
    QtConcurrent::run(this, &Classifier::_classify, file);
}

/*** PRIVATE SLOTS ***/
void Classifier::setClassification(const QString &key, const QString &value)
{
    if (key.isEmpty()) clear();
    else               setText(QString("<b>%1: </b>%2").arg(key, value));
}

/*** PRIVATE ***/
void Classifier::_classify(File file)
{
    QString key, value;
    foreach (const File &f, Enroll(file.flat(), File("[algorithm=" + algorithm + "]"))) {
        if (!f.contains("Label"))
            continue;

        if (algorithm == "GenderClassification") {
            key = "Gender";
            value = f.get<QString>(key);
        } else if (algorithm == "AgeRegression") {
            key = "Age";
            value = QString::number(int(f.get<float>(key)+0.5)) + " Years";
        } else {
            key = algorithm;
            value = f.get<QString>("Label");
        }
        break;
    }

    emit newClassification(key, value);
}

#include "moc_classifier.cpp"
