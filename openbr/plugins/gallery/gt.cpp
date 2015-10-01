#include "openbr/plugins/openbr_internal.h"
#include "openbr/core/qtutils.h"

using namespace br;

/*!
 * \ingroup galleries
 * \brief Ground truth format for evaluating kNN
 * \author Josh Klontz \cite jklontz
 */
class gtGallery : public Gallery
{
    Q_OBJECT

    TemplateList templates;

    ~gtGallery()
    {
        const QList<int> labels = File::get<int>(TemplateList::relabel(templates, "Label", false), "Label");

        QStringList lines;
        for (int i=0; i<labels.size(); i++) {
            QStringList words;
            for (int j=0; j<labels.size(); j++) {
                if ((labels[i] == labels[j]) && (i != j))
                    words.append(QString::number(j));
            }
            lines.append(words.join("\t"));
        }

        QtUtils::writeFile(file.name, lines);
    }

    TemplateList readBlock(bool *done)
    {
        *done = true;
        qFatal("Not supported!");
        return TemplateList();
    }

    void write(const Template &t)
    {
        templates.append(t);
    }
};

BR_REGISTER(Gallery, gtGallery)

#include "gt.moc"
