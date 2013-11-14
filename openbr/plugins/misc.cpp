/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2012 The MITRE Corporation                                      *
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License");           *
 * you may not use this file except in compliance with the License.          *
 * You may obtain a copy of the License at                                   *
 *                                                                           *
 *     http://www.apache.org/licenses/LICENSE-2.0                            *
 *                                                                           *
 * Unless required by applicable law or agreed to in writing, software       *
 * distributed under the License is distributed on an "AS IS" BASIS,         *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 * See the License for the specific language governing permissions and       *
 * limitations under the License.                                            *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <QElapsedTimer>
#include <QRegularExpression>
#include <opencv2/highgui/highgui.hpp>
#include "openbr_internal.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/qtutils.h"

using namespace cv;

namespace br
{

/*!
 * \ingroup transforms
 * \brief Applies br::Format to br::Template::file::name and appends results.
 * \author Josh Klontz \cite jklontz
 */
class OpenTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        if (!src.isEmpty()) { dst = src; return; }
        if (Globals->verbose) qDebug("Opening %s", qPrintable(src.file.flat()));
        dst.file = src.file;
        foreach (const File &file, src.file.split()) {
            QScopedPointer<Format> format(Factory<Format>::make(file));
            Template t = format->read();
            if (t.isEmpty()) qWarning("Can't open %s from %s", qPrintable(file.flat()), qPrintable(QDir::currentPath()));
            dst.append(t);
            dst.file.append(t.file.localMetadata());
        }
        dst.file.set("FTO", dst.isEmpty());
    }
};

BR_REGISTER(Transform, OpenTransform)

/*!
 * \ingroup transforms
 * \brief Prints the template's file to stdout or stderr.
 * \author Josh Klontz \cite jklontz
 */
class PrintTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(bool error READ get_error WRITE set_error RESET reset_error)
    Q_PROPERTY(bool data READ get_data WRITE set_data RESET reset_data)
    BR_PROPERTY(bool, error, true)
    BR_PROPERTY(bool, data, false)

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        const QString nameString = src.file.flat();
        const QString dataString = data ? OpenCVUtils::matrixToString(src)+"\n" : QString();
        QStringList matricies;
        foreach (const Mat &m, src)
            matricies.append(QString::number(m.rows) + "x" + QString::number(m.cols) + "_" + OpenCVUtils::typeToString(m));
        fprintf(error ? stderr : stdout, "%s\n  %s\n%s", qPrintable(nameString), qPrintable(matricies.join(",")), qPrintable(dataString));
    }
};

BR_REGISTER(Transform, PrintTransform)

/*!
 * \ingroup transforms
 * \brief Checks the template for NaN values.
 * \author Josh Klontz \cite jklontz
 */
class CheckTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    static int count;
    int index;

 public:
    CheckTransform() : index(count++) {}

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        foreach (const Mat &m, src) {
            Mat fm;
            m.convertTo(fm, CV_32F);
            const int elements = fm.rows * fm.cols * fm.channels();
            const float *data = (const float*)fm.data;
            for (int i=0; i<elements; i++)
                if (data[i] != data[i])
                    qFatal("%s NaN check %d failed!", qPrintable(src.file.flat()), index);
        }
    }
};

int CheckTransform::count = 0;

BR_REGISTER(Transform, CheckTransform)

/*!
 * \ingroup transforms
 * \brief Sets the template's matrix data to the br::File::name.
 * \author Josh Klontz \cite jklontz
 */
class NameTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;
        QByteArray name = src.file.baseName().toLocal8Bit();
        dst = Mat(1, name.size(), CV_8UC1);
        memcpy(dst.m().data, name.data(), name.size());
    }
};

BR_REGISTER(Transform, NameTransform)

/*!
 * \ingroup transforms
 * \brief A no-op transform.
 * \see DiscardTransform FirstTransform RestTransform RemoveTransform
 * \author Josh Klontz \cite jklontz
 */
class IdentityTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst = src;
    }
};

BR_REGISTER(Transform, IdentityTransform)

/*!
 * \ingroup transforms
 * \brief Removes all template's matrices.
 * \see IdentityTransform FirstTransform RestTransform RemoveTransform
 * \author Josh Klontz \cite jklontz
 */
class DiscardTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;
    }
};

BR_REGISTER(Transform, DiscardTransform)

/*!
 * \ingroup transforms
 * \brief Removes all but the first matrix from the template.
 * \see IdentityTransform DiscardTransform RestTransform RemoveTransform
 * \author Josh Klontz \cite jklontz
 */
class FirstTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        // AggregateFrames will leave the Template empty
        // if it hasn't filled up the buffer
        // so we gotta anticipate an empty Template
        if (src.empty()) return;
        dst.file = src.file;
        dst = src.m();
    }
};

BR_REGISTER(Transform, FirstTransform)

/*!
 * \ingroup transforms
 * \brief Removes the first matrix from the template.
 * \see IdentityTransform DiscardTransform FirstTransform RemoveTransform
 * \author Josh Klontz \cite jklontz
 */
class RestTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        dst.removeFirst();
    }
};

BR_REGISTER(Transform, RestTransform)

/*!
 * \ingroup transforms
 * \brief Removes the matrix from the template at the specified index.
 * \author Josh Klontz \cite jklontz
 * \see IdentityTransform DiscardTransform FirstTransform RestTransform
 */
//! [example_transform]
class RemoveTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(int index READ get_index WRITE set_index RESET reset_index STORED false)
    BR_PROPERTY(int, index, 0)

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        dst.removeAt(index);
    }
};

BR_REGISTER(Transform, RemoveTransform)
//! [example_transform]

/*!
 * \ingroup transforms
 * \brief Rename metadata key
 * \author Josh Klontz \cite jklontz
 */
class RenameTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString find READ get_find WRITE set_find RESET reset_find STORED false)
    Q_PROPERTY(QString replace READ get_replace WRITE set_replace RESET reset_replace STORED false)
    BR_PROPERTY(QString, find, "")
    BR_PROPERTY(QString, replace, "")

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        if (dst.file.localKeys().contains(find)) {
            dst.file.set(replace, dst.file.value(find));
            dst.file.remove(find);
        }
    }
};

BR_REGISTER(Transform, RenameTransform)

/*!
 * \ingroup transforms
 * \brief Rename first found metadata key
 * \author Josh Klontz \cite jklontz
 */
class RenameFirstTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QStringList find READ get_find WRITE set_find RESET reset_find STORED false)
    Q_PROPERTY(QString replace READ get_replace WRITE set_replace RESET reset_replace STORED false)
    BR_PROPERTY(QStringList, find, QStringList())
    BR_PROPERTY(QString, replace, "")

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        foreach (const QString &key, find)
            if (dst.file.localKeys().contains(key)) {
                dst.file.set(replace, dst.file.value(key));
                dst.file.remove(key);
                break;
            }
    }
};

BR_REGISTER(Transform, RenameFirstTransform)

/*!
 * \ingroup transforms
 * \brief Name a point
 * \author Scott Klum \cite sklum
 */
class LabelTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QList<int> points READ get_points WRITE set_points RESET reset_points STORED false)
    Q_PROPERTY(QStringList names READ get_names WRITE set_names RESET reset_names STORED false)
    BR_PROPERTY(QList<int>, points, QList<int>())
    BR_PROPERTY(QStringList, names, QStringList())

    void project(const Template &src, Template &dst) const
    {
        if (points.size() != names.size()) qFatal("Point/name size mismatch");

        dst = src;

        for (int i=0; i<points.size(); i++)
            dst.file.set(names[i], points[i]);
    }
};

BR_REGISTER(Transform, LabelTransform)

/*!
 * \ingroup transforms
 * \brief Remove a name from a point
 * \author Scott Klum \cite sklum
 */
class AnonymizeTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QStringList names READ get_names WRITE set_names RESET reset_names STORED false)
    BR_PROPERTY(QStringList, names, QStringList())

    void project(const Template &src, Template &dst) const
    {
        dst = src;

        foreach (const QString &name, names)
            if (src.file.contains(name)) dst.file.appendPoint(src.file.get<QPointF>(name));
    }
};

BR_REGISTER(Transform, AnonymizeTransform)

/*!
 * \ingroup transforms
 * \brief Change the br::Template::file extension
 * \author Josh Klontz \cite jklontz
 */
class AsTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString extension READ get_extension WRITE set_extension RESET reset_extension STORED false)
    BR_PROPERTY(QString, extension, "")

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        dst.file.name = dst.file.name.left(dst.file.name.lastIndexOf('.')+1) + extension;
    }
};

BR_REGISTER(Transform, AsTransform)

/*!
 * \ingroup transforms
 * \brief Apply the input regular expression to the value of inputProperty, store the matched portion in outputProperty.
 * \author Charles Otto \cite caotto
 */
class RegexPropertyTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString regexp READ get_regexp WRITE set_regexp RESET reset_regexp STORED false)
    Q_PROPERTY(QString inputProperty READ get_inputProperty WRITE set_inputProperty RESET reset_inputProperty STORED false)
    Q_PROPERTY(QString outputProperty READ get_outputProperty WRITE set_outputProperty RESET reset_outputProperty STORED false)
    BR_PROPERTY(QString, regexp, "(.*)")
    BR_PROPERTY(QString, inputProperty, "name")
    BR_PROPERTY(QString, outputProperty, "Label")

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        QRegularExpression re(regexp);
        QRegularExpressionMatch match = re.match(dst.file.get<QString>(inputProperty));
        if (!match.hasMatch())
            qFatal("Unable to match regular expression \"%s\" to base name \"%s\"!", qPrintable(regexp), qPrintable(dst.file.get<QString>(inputProperty)));
        dst.file.set(outputProperty, match.captured(match.lastCapturedIndex()));
    }
};

BR_REGISTER(Transform, RegexPropertyTransform)

/*!
 * \ingroup transforms
 * \brief Store the last matrix of the input template as a metadata key with input property name.
 * \author Charles Otto \cite caotto
 */
class SaveMatTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    Q_PROPERTY(QString propName READ get_propName WRITE set_propName RESET reset_propName STORED false)
    BR_PROPERTY(QString, propName, "")

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        dst.file.set(propName, QVariant::fromValue(dst.m()));
    }
};
BR_REGISTER(Transform, SaveMatTransform)

/*!
 * \ingroup transforms
 * \brief Set the last matrix of the input template to a matrix stored as metadata with input propName.
 *
 * Also removes the property from the templates metadata after restoring it.
 *
 * \author Charles Otto \cite caotto
 */
class RestoreMatTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString propName READ get_propName WRITE set_propName RESET reset_propName STORED false)
    BR_PROPERTY(QString, propName, "")

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        if (dst.file.contains(propName)) {
            dst.clear();
            dst.m() = dst.file.get<cv::Mat>(propName);
            dst.file.remove(propName);
        }
    }
};
BR_REGISTER(Transform, RestoreMatTransform)


/*!
 * \ingroup transforms
 * \brief Incrementally output templates received to a gallery, based on the current filename
 * When a template is received in projectUpdate for the first time since a finalize, open a new gallery based on the
 * template's filename, and the galleryFormat property.
 * Templates received in projectUpdate will be output to the gallery with a filename combining their original filename and
 * their FrameNumber property, with the file extension specified by the fileFormat property.
 * \author Charles Otto \cite caotto
 */
class IncrementalOutputTransform : public TimeVaryingTransform
{
    Q_OBJECT

    Q_PROPERTY(QString galleryFormat READ get_galleryFormat WRITE set_galleryFormat RESET reset_galleryFormat STORED false)
    Q_PROPERTY(QString fileFormat READ get_fileFormat WRITE set_fileFormat RESET reset_fileFormat STORED false)
    BR_PROPERTY(QString, galleryFormat, "")
    BR_PROPERTY(QString, fileFormat, ".png")

    bool galleryUp;

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        if (src.empty())
            return;

        if (!galleryUp) {
            QFileInfo finfo(src[0].file.name);
            QString galleryName = finfo.baseName() + galleryFormat;

            writer = QSharedPointer<Gallery> (Factory<Gallery>::make(galleryName));
            galleryUp = true;
        }

        dst = src;
        int idx =0;
        foreach(const Template & t, src) {
            if (t.empty())
                continue;

            // Build the output filename for this template
            QFileInfo finfo(t.file.name);
            QString outputName = finfo.baseName() +"_" + t.file.get<QString>("FrameNumber") + "_" + QString::number(idx)+ fileFormat;

            idx++;
            Template out = t;
            out.file.name = outputName;
            writer->write(out);
        }
    }

    void train(const TemplateList& data)
    {
        (void) data;
    }

    // Drop the current gallery.
    void finalize(TemplateList & data)
    {
        (void) data;
        galleryUp = false;
    }

    QSharedPointer<Gallery> writer;
public:
    IncrementalOutputTransform() : TimeVaryingTransform(false,false) {galleryUp = false;}
};

BR_REGISTER(Transform, IncrementalOutputTransform)

class EventTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString eventName READ get_eventName WRITE set_eventName RESET reset_eventName STORED false)
    BR_PROPERTY(QString, eventName, "")

    TemplateEvent event;

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        event.pulseSignal(dst);
    }

    TemplateEvent * getEvent(const QString & name)
    {
        return name == eventName ? &event : NULL;
    }
};
BR_REGISTER(Transform, EventTransform)


class GalleryOutputTransform : public TimeVaryingTransform
{
    Q_OBJECT

    Q_PROPERTY(QString outputString READ get_outputString WRITE set_outputString RESET reset_outputString STORED false)
    BR_PROPERTY(QString, outputString, "")

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        if (src.empty())
            return;
        dst = src;
        writer->writeBlock(dst);
    }

    void train(const TemplateList& data)
    {
        (void) data;
    }
    ;
    void init()
    {
        writer = QSharedPointer<Gallery>(Gallery::make(outputString));
    }

    QSharedPointer<Gallery> writer;
public:
    GalleryOutputTransform() : TimeVaryingTransform(false,false) {}
};

BR_REGISTER(Transform, GalleryOutputTransform)

class ProgressCounterTransform : public TimeVaryingTransform
{
    Q_OBJECT

    Q_PROPERTY(int totalTemplates READ get_totalTemplates WRITE set_totalTemplates RESET reset_totalTemplates STORED false)
    BR_PROPERTY(int, totalTemplates, 1)

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        dst = src;
        qint64 elapsed = timer.elapsed();
        calls++;
        set_calls++;
        // updated every 10 seconds
        if (elapsed > 5 * 1000) {
            float f_elapsed = elapsed / 1000.0f;
            // remaining calls (according to our input variable)
            int remaining = totalTemplates - calls;
            // calls / second
            float speed = set_calls / f_elapsed;

            float p = 100 * float(calls) / totalTemplates;

            // seconds remaining
            int s = float(remaining) / speed;

            fprintf(stderr, "%05.2f%%  ELAPSED=%s  REMAINING=%s  COUNT=%g  \r", p, QtUtils::toTime(Globals->startTime.elapsed()/1000.0f).toStdString().c_str(), QtUtils::toTime(s).toStdString().c_str(), float(calls));

            timer.start();
            set_calls = 0;
        }


        return;
    }

    void train(const TemplateList& data)
    {
        (void) data;
    }

    void finalize(TemplateList & data)
    {
        (void) data;
        float p = 100 * float(calls) / totalTemplates;
        qDebug("%05.2f%%  ELAPSED=%s  REMAINING=%s  COUNT=%g  \r", p, QtUtils::toTime(Globals->startTime.elapsed()/1000.0f).toStdString().c_str(), QtUtils::toTime(0).toStdString().c_str(), float(calls));
    }

    void init()
    {
        calls = 0;
        set_calls = 0;
        timer.start();
        Globals->startTime.start();
    }

public:
    ProgressCounterTransform() : TimeVaryingTransform(false,false) {}
    bool initialized;
    QElapsedTimer timer;
    qint64 calls;
    qint64 set_calls;

};

BR_REGISTER(Transform, ProgressCounterTransform)

}

#include "misc.moc"
