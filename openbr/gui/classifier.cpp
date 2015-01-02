#include <QtConcurrentRun>
#include <openbr/openbr_plugin.h>

#include "classifier.h"

using namespace br;

/**** CLASSIFIER ****/
/*** PUBLIC ***/
GUIClassifier::GUIClassifier(QWidget *parent)
    : QLabel(parent)
{
    setAlignment(Qt::AlignCenter);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    connect(this, SIGNAL(newClassification(QString,QString)), this, SLOT(setClassification(QString,QString)));
}

void GUIClassifier::setAlgorithm(const QString &algorithm)
{
    this->algorithm = algorithm;
}

/*** PUBLIC SLOTS ***/
void GUIClassifier::classify(const File &file)
{
    QtConcurrent::run(this, &GUIClassifier::_classify, file);
}

/*** PRIVATE SLOTS ***/
void GUIClassifier::setClassification(const QString &key, const QString &value)
{
    if (key.isEmpty()) clear();
    else               setText(QString("%1: <b>%2</b>").arg(key, value));
}

/*** PRIVATE ***/
void GUIClassifier::_classify(File file)
{
    QString key, value;
    QSharedPointer<Transform> transform = Transform::fromAlgorithm(algorithm);

    TemplateList input, output;
    input.append(file);
    transform->projectUpdate(input, output);

    foreach (const File &f, output.files() ) {
        if      (algorithm == "GenderClassification") key = "Gender";
        else if (algorithm == "AgeRegression")        key = "Age";
        else                                          key = algorithm;

        if (!f.contains(key)) continue;

        if (algorithm == "AgeRegression") value = QString::number(int(f.get<float>(key)+0.5)) + " Years";
        else                              value = f.get<QString>(key);

        break;
    }

    emit newClassification(key, value);
}

#include "moc_classifier.cpp"
