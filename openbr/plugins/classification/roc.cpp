#include <QDebug>
#include <QFileInfo>
#include <QMap>
#include <QMutex>
#include <QSharedPointer>
#include <QString>
#include <QStringList>
#include <QThreadPool>
#include <QVariant>
#include <roc.h>
#include "openbr/plugins/openbr_internal.h"
#include "openbr/core/resource.h"

using namespace br;

class ROCInitializer : public Initializer
{
    Q_OBJECT

    void initialize() const
    {
	ROC_ENSURE(roc_initialize(qPrintable(Globals->sdkPath)))
    }

    void finalize() const
    {
        ROC_ENSURE(roc_finalize())
    }
};


#include "classification/roc.moc"
