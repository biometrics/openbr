#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/qtutils.h>

namespace br
{

/*!
 * \ingroup galleries
 * \brief Implements the FDDB detection format.
 * \author Josh Klontz \cite jklontz
 *
 * http://vis-www.cs.umass.edu/fddb/README.txt
 */
class FDDBGallery : public Gallery
{
    Q_OBJECT

    TemplateList readBlock(bool *done)
    {
        *done = true;
        QStringList lines = QtUtils::readLines(file);
        TemplateList templates;
        while (!lines.empty()) {
            const QString fileName = lines.takeFirst();
            int numDetects = lines.takeFirst().toInt();
            for (int i=0; i<numDetects; i++) {
                const QStringList detect = lines.takeFirst().split(' ');
                Template t(fileName);
                QList<QVariant> faceList; //to be consistent with slidingWindow
                if (detect.size() == 5) { //rectangle
                    faceList.append(QRectF(detect[0].toFloat(), detect[1].toFloat(), detect[2].toFloat(), detect[3].toFloat()));
                    t.file.set("Confidence", detect[4].toFloat());
                } else if (detect.size() == 6) { //ellipse
                    float x = detect[3].toFloat(),
                          y = detect[4].toFloat(),
                          radius = detect[1].toFloat();
                    faceList.append(QRectF(x - radius,y - radius,radius * 2.0, radius * 2.0));
                    t.file.set("Confidence", detect[5].toFloat());
                } else {
                    qFatal("Unknown FDDB annotation format.");
                }
                t.file.set("Face", faceList);
                t.file.set("Label",QString("face"));
                templates.append(t);
            }
        }
        return templates;
    }

    void write(const Template &t)
    {
        (void) t;
        qFatal("Not implemented.");
    }

    void init()
    {
        //
    }
};

BR_REGISTER(Gallery, FDDBGallery)

} // namespace br

#include "gallery/fddb.moc"
