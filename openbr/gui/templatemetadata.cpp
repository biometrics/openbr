#include <QAction>
#include <QFileInfo>

#include "templatemetadata.h"

using namespace br;

/*** PUBLIC ***/
TemplateMetadata::TemplateMetadata(QWidget *parent)
    : QToolBar("Template Metadata", parent)
{
    lFile.setTextInteractionFlags(Qt::TextSelectableByMouse);
    wSpacer.setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    addWidget(&wOffset);
    addWidget(&lFile);
    addWidget(&wSpacer);
    addWidget(&lQuality);
}

void TemplateMetadata::addClassifier(const QString &classifier_, const QString algorithm)
{
    QSharedPointer<Classifier> classifier(new Classifier());
    classifier->setAlgorithm(classifier_);
    QAction *action = addWidget(classifier.data());
    conditionalClassifiers.append(ConditionalClassifier(algorithm, classifier, action));
}

/*** PUBLIC SLOTS ***/
void TemplateMetadata::setFile(const File &file)
{
    if (file.isNull()) lFile.clear();
    else               lFile.setText("<b>File:</b> " + file.fileName());
    lQuality.setText(QString("<b>Quality:</b> %1").arg(file.get<bool>("FTE", false) ? "Low" : "High"));
    foreach (const ConditionalClassifier &classifier, conditionalClassifiers)
        if (classifier.action->isVisible()) classifier.classifier->classify(file);
}

void TemplateMetadata::setAlgorithm(const QString &algorithm)
{
    foreach (const ConditionalClassifier &classifier, conditionalClassifiers) {
        classifier.classifier->clear();
        classifier.action->setVisible(classifier.algorithm.isEmpty() || classifier.algorithm == algorithm);
    }
}

#include "moc_templatemetadata.cpp"
