#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/qtutils.h>

namespace br
{

/*!
 * \ingroup outputs
 * \brief One score per row.
 * \author Josh Klontz \cite jklontz
 */
class meltOutput : public MatrixOutput
{
    Q_OBJECT

    ~meltOutput()
    {
        if (file.isNull() || targetFiles.isEmpty() || queryFiles.isEmpty()) return;
        const bool genuineOnly = file.contains("Genuine") && !file.contains("Impostor");
        const bool impostorOnly = file.contains("Impostor") && !file.contains("Genuine");

        QMap<QString,QVariant> args = file.localMetadata();
        args.remove("Genuine");
        args.remove("Impostor");

        QString keys; foreach (const QString &key, args.keys()) keys += "," + key;
        QString values; foreach (const QVariant &value, args.values()) values += "," + value.toString();

        QStringList lines;
        if (file.baseName() != "terminal") lines.append(QString("Query,Target,Mask,Similarity%1").arg(keys));

        QList<QString> queryLabels = File::get<QString>(queryFiles, "Label");
        QList<QString> targetLabels = File::get<QString>(targetFiles, "Label");

        for (int i=0; i<queryFiles.size(); i++) {
            for (int j=(selfSimilar ? i+1 : 0); j<targetFiles.size(); j++) {
                const bool genuine = queryLabels[i] == targetLabels[j];
                if ((genuineOnly && !genuine) || (impostorOnly && genuine)) continue;
                lines.append(QString("%1,%2,%3,%4%5").arg(queryFiles[i],
                                                          targetFiles[j],
                                                          QString::number(genuine),
                                                          QString::number(data.at<float>(i,j)),
                                                          values));
            }
        }

        QtUtils::writeFile(file, lines);
    }
};

BR_REGISTER(Output, meltOutput)

} // namespace br

#include "output/melt.moc"
