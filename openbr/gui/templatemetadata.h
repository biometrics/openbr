#ifndef BR_TEMPLATEMETADATA_H
#define BR_TEMPLATEMETADATA_H

#include <QBoxLayout>
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

class BR_EXPORT TemplateMetadata : public QWidget
{
    Q_OBJECT
    QHBoxLayout *layout;
    QLabel *lFile, *lQuality;

    struct ConditionalClassifier
    {
        QString algorithm;
        QSharedPointer<GUIClassifier> classifier;

        ConditionalClassifier() {}
        ConditionalClassifier(const QString &algorithm_, const QSharedPointer<GUIClassifier> &classifier_)
            : algorithm(algorithm_), classifier(classifier_) {}
    };
    QList<ConditionalClassifier> conditionalClassifiers;

public:
    explicit TemplateMetadata(QWidget *parent = 0);
    void addClassifier(const QString &classifier, const QString algorithm = "");

public slots:
    void setFile(const File &file);
    void setAlgorithm(const QString &algorithm);
    void showQuality(bool visible);
};

} // namespace br

#endif // BR_TEMPLATEMETADATA_H
