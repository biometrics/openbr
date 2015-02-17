#include <QtConcurrent>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/common.h>
#include <openbr/core/eval.h>

namespace br
{

/*!
 * \ingroup outputs
 * \brief Evaluate the output matrix.
 * \author Josh Klontz \cite jklontz
 */
class evalOutput : public MatrixOutput
{
    Q_OBJECT
    Q_PROPERTY(QString target READ get_target WRITE set_target RESET reset_target STORED false)
    Q_PROPERTY(QString query READ get_query WRITE set_query RESET reset_query STORED false)
    Q_PROPERTY(bool crossValidate READ get_crossValidate WRITE set_crossValidate RESET reset_crossValidate STORED false)
    BR_PROPERTY(bool, crossValidate, true)
    BR_PROPERTY(QString, target, QString())
    BR_PROPERTY(QString, query, QString())

    ~evalOutput()
    {
        if (!target.isEmpty()) targetFiles = TemplateList::fromGallery(target).files();
        if (!query.isEmpty())  queryFiles  = TemplateList::fromGallery(query).files();

        if (data.data) {
            const QString csv = QString(file.name).replace(".eval", ".csv");
            if ((Globals->crossValidate == 0) || (!crossValidate)) {
                Evaluate(data, targetFiles, queryFiles, csv);
            } else {
                QFutureSynchronizer<float> futures;
                for (int i=0; i<Globals->crossValidate; i++)
                    futures.addFuture(QtConcurrent::run(Evaluate, data, targetFiles, queryFiles, csv.arg(QString::number(i)), i));
                futures.waitForFinished();

                QList<float> TARs;
                foreach (const QFuture<float> &future, futures.futures())
                    TARs.append(future.result());

                double mean, stddev;
                Common::MeanStdDev(TARs, &mean, &stddev);
                qDebug("TAR @ FAR = 0.01: %.3f +/- %.3f", mean, stddev);
            }
        }
    }
};

BR_REGISTER(Output, evalOutput)

} // namespace br

#include "output/eval.moc"
