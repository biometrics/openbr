#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/qtutils.h>

namespace br
{

/*!
 * \ingroup outputs
 * \brief Score histogram.
 * \author Josh Klontz \cite jklontz
 */
class histOutput : public Output
{
    Q_OBJECT

    float min, max, step;
    QVector<int> bins;

    ~histOutput()
    {
        if (file.isNull() || bins.isEmpty()) return;
        QStringList counts;
        foreach (int count, bins)
            counts.append(QString::number(count));
        const QString result = counts.join(",");
        QtUtils::writeFile(file, result);
    }

    void initialize(const FileList &targetFiles, const FileList &queryFiles)
    {
        Output::initialize(targetFiles, queryFiles);
        min = file.get<float>("min", -5);
        max = file.get<float>("max", 5);
        step = file.get<float>("step", 0.1);
        bins = QVector<int>((max-min)/step, 0);
    }

    void set(float value, int i, int j)
    {
        (void) i;
        (void) j;
        if ((value < min) || (value >= max)) return;
        bins[(value-min)/step]++; // This should technically be locked to ensure atomic increment
    }
};

BR_REGISTER(Output, histOutput)

} // namespace br

#include "output/hist.moc"
