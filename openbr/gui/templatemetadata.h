#ifndef BR_TEMPLATEMETADATA_H
#define BR_TEMPLATEMETADATA_H

#include <QLabel>
#include <QList>
#include <QPair>
#include <QSharedPointer>
#include <QToolBar>
#include <QWidget>
#include <openbr/openbr_plugin.h>

#include "classifier.h"

namespace br
{

class BR_EXPORT TemplateMetadata : public QToolBar
{
    Q_OBJECT
    QLabel lFile, lQuality;
    QWidget wOffset, wSpacer;

    struct ConditionalClassifier
    {
        QString algorithm;
        QSharedPointer<Classifier> classifier;
        QAction *action;

        ConditionalClassifier() : action(NULL) {}
        ConditionalClassifier(const QString &algorithm_, const QSharedPointer<Classifier> &classifier_, QAction *action_)
            : algorithm(algorithm_), classifier(classifier_), action(action_) {}
    };
    QList<ConditionalClassifier> conditionalClassifiers;

public:
    explicit TemplateMetadata(QWidget *parent = 0);
    void addClassifier(const QString &classifier, const QString algorithm = "");

public slots:
    void setFile(const File &file);
    void setAlgorithm(const QString &algorithm);
};

} // namespace br

#endif // BR_TEMPLATEMETADATA_H
