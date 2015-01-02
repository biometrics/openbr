#include <QAction>
#include <QFileInfo>

#include "templatemetadata.h"

using namespace br;

/*** PUBLIC ***/
TemplateMetadata::TemplateMetadata(QWidget *parent)
    : QWidget(parent)
{
    layout = new QHBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    lFile = new QLabel(this);
    lFile->setTextInteractionFlags(Qt::TextSelectableByMouse);
    lQuality = new QLabel(this);
    layout->addWidget(lFile);
    layout->addWidget(lQuality);
    showQuality(false);
}

void TemplateMetadata::addClassifier(const QString &classifier_, const QString algorithm)
{
    QSharedPointer<GUIClassifier> classifier(new GUIClassifier());
    classifier->setAlgorithm(classifier_);
    layout->addWidget(classifier.data());
    conditionalClassifiers.append(ConditionalClassifier(algorithm, classifier));
}

/*** PUBLIC SLOTS ***/
void TemplateMetadata::setFile(const File &file)
{
    if (file.isNull()) lFile->clear();
    else               lFile->setText("File: <b>" + file.fileName() + "</b>");
    lQuality->setText(QString("Quality: <b>%1</b>").arg(file.get<bool>("FTE", false) ? "Low" : "High"));
    foreach (const ConditionalClassifier &classifier, conditionalClassifiers)
        if (classifier.classifier->isVisible()) classifier.classifier->classify(file);
}

void TemplateMetadata::setAlgorithm(const QString &algorithm)
{
    foreach (const ConditionalClassifier &classifier, conditionalClassifiers) {
        classifier.classifier->clear();
        classifier.classifier->setVisible(classifier.algorithm.isEmpty() || classifier.algorithm == algorithm);
    }
}

void TemplateMetadata::showQuality(bool visible)
{
    lQuality->setVisible(visible);
}

#include "moc_templatemetadata.cpp"
