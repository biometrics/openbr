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

#include <QFutureSynchronizer>
#include <QMetaProperty>
#include <QPointF>
#include <QRect>
#include <QRegExp>
#include <QSettings>
#include <QThreadPool>
#include <QtConcurrentRun>
#ifdef BR_DISTRIBUTED
#include <mpi.h>
#endif // BR_DISTRIBUTED
#include <algorithm>
#include <iostream>
#include <openbr/openbr_plugin.h>

#ifndef BR_EMBEDDED
#include <QApplication>
#endif

#include "version.h"
#include "core/bee.h"
#include "core/common.h"
#include "core/opencvutils.h"
#include "core/qtutils.h"

using namespace br;
using namespace cv;

/* File - public methods */
// Note that the convention for displaying metadata is as follows:
// [] for lists in which argument order does not matter (e.g. [FTO=false, Index=0]),
// () for lists in which argument order matters (e.g. First_Eye(100.0,100.0)).
QString File::flat() const
{
    QStringList values;
    QStringList keys = this->localKeys(); qSort(keys);
    foreach (const QString &key, keys) {
        const QVariant value = this->value(key);
        if (value.isNull()) values.append(key);
        else {
            if (QString(value.typeName()) == "QVariantList") {
                QStringList landmarks;
                foreach(const QVariant &landmark, qvariant_cast<QVariantList>(value)) {
                    landmarks.append(toString(landmark));
                }
                if (!landmarks.isEmpty()) values.append(key + "=[" + landmarks.join(", ") + "]");
            }
            else values.append(key + "=" + toString(value));
        }
    }

    QString flat = name;
    if (!values.isEmpty()) flat += "[" + values.join(", ") + "]";
    return flat;
}

QString File::hash() const
{
    return QtUtils::shortTextHash(flat());
}

void File::append(const QMap<QString,QVariant> &metadata)
{
    foreach (const QString &key, metadata.keys())
        set(key, metadata[key]);
}

void File::append(const File &other)
{
    if (!other.name.isEmpty() && name != other.name) {
        if (name.isEmpty()) {
            name = other.name;
        } else {
            if (!contains("separator")) set("separator", ";");
            name += value("separator").toString() + other.name;
        }
    }
    append(other.m_metadata);
}

QList<File> File::split() const
{
    if (name.isEmpty()) return QList<File>();
    if (!contains("separator")) return QList<File>() << *this;
    return split(value("separator").toString());
}

QList<File> File::split(const QString &separator) const
{
    QList<File> files;
    foreach (const QString &word, name.split(separator, QString::SkipEmptyParts)) {
        File file(word);
        file.append(m_metadata);
        files.append(file);
    }
    return files;
}

QString File::resolved() const
{
    return exists() ? name : Globals->path + "/" + name;
}

bool File::contains(const QString &key) const
{
    return m_metadata.contains(key) || Globals->contains(key);
}

QVariant File::value(const QString &key) const
{
    return m_metadata.contains(key) ? m_metadata.value(key) : Globals->property(qPrintable(key));
}

void File::set(const QString &key, const QVariant &value)
{
    if (key == "Label") {
        bool ok = false;
        const QString valueString = value.toString();

        /* We assume that if the value starts with '0'
           then it was probably intended to to be a string UID
           and that it's numerical value is not relevant. */
        if (value.canConvert(QVariant::Double) &&
            (!valueString.startsWith('0') || (valueString == "0")))
            value.toFloat(&ok);

        if (!ok && !Globals->classes.contains(valueString))
            Globals->classes.insert(valueString, Globals->classes.size());
    }

    m_metadata.insert(key, value);
}

bool File::getBool(const QString &key) const
{
    if (!contains(key)) return false;
    QVariant variant = value(key);
    if (variant.isNull() || !variant.canConvert<bool>()) return true;
    return variant.value<bool>();
}

QString File::subject(int label)
{
    return Globals->classes.key(label, QString::number(label));
}

float File::label() const
{
    const QVariant variant = value("Label");
    if (variant.isNull()) return -1;

    if (Globals->classes.contains(variant.toString()))
        return Globals->classes.value(variant.toString());

    bool ok;
    const float val = variant.toFloat(&ok);
    return ok ? val : -1;
}

QList<QPointF> File::namedPoints() const
{
    QList<QPointF> landmarks;
    foreach (const QString &key, localMetadata().keys()) {
        const QVariant &variant = m_metadata[key];
        if (variant.canConvert<QPointF>())
            landmarks.append(variant.value<QPointF>());
    }
    return landmarks;
}

QList<QPointF> File::points() const
{
    QList<QPointF> points;
    foreach (const QVariant &point, m_metadata["Points"].toList())
        points.append(point.toPointF());
    return points;
}

void File::appendPoint(const QPointF &point)
{
    QList<QVariant> newPoints = m_metadata["Points"].toList();
    newPoints.append(point);
    m_metadata["Points"] = newPoints;
}

void File::appendPoints(const QList<QPointF> &points)
{
    QList<QVariant> newPoints = m_metadata["Points"].toList();
    foreach (const QPointF &point, points)
        newPoints.append(point);
    m_metadata["Points"] = newPoints;
}

QList<QRectF> File::namedRects() const
{
    QList<QRectF> rects;
    foreach (const QString &key, localMetadata().keys()) {
        const QVariant &variant = m_metadata[key];
        if (variant.canConvert<QRectF>())
            rects.append(variant.value<QRectF>());
    }
    return rects;
}

QList<QRectF> File::rects() const
{
    QList<QRectF> rects;
    foreach (const QVariant &rect, m_metadata["Rects"].toList())
        rects.append(rect.toRect());
    return rects;
}

void File::appendRect(const QRectF &rect)
{
    QList<QVariant> newRects = m_metadata["Rects"].toList();
    newRects.append(rect);
    m_metadata["Rects"] = newRects;
}

void File::appendRects(const QList<QRectF> &rects)
{
    QList<QVariant> newRects = m_metadata["Rects"].toList();
    foreach (const QRectF &rect, rects)
        newRects.append(rect);
    m_metadata["Rects"] = newRects;
}

/* File - private methods */
void File::init(const QString &file)
{
    name = file;

    while (name.endsWith(']') || name.endsWith(')')) {
        const bool unnamed = name.endsWith(')');

        int index, depth = 0;
        for (index = name.size()-1; index >= 0; index--) {
            if      (name[index] == (unnamed ? ')' : ']')) depth--;
            else if (name[index] == (unnamed ? '(' : '[')) depth++;
            if (depth == 0) break;
        }
        if (depth != 0) qFatal("Unable to parse: %s", qPrintable(file));

        const QStringList parameters = QtUtils::parse(name.mid(index+1, name.size()-index-2));
        for (int i=0; i<parameters.size(); i++) {
            QStringList words = QtUtils::parse(parameters[i], '=');
            QtUtils::checkArgsSize("File", words, 1, 2);
            if (words.size() < 2) {
                if (unnamed) setParameter(i, words[0]);
                else         set(words[0], QVariant());
            } else {
                fromString(words[0],words[1]);
            }
        }
        name = name.left(index);
    }
}

QString File::toString(const QVariant &variant) const
{
    if (variant.canConvert(QVariant::String)) return variant.toString();
    else if(variant.canConvert(QVariant::PointF)) return QString("(%1,%2)").arg(QString::number(qvariant_cast<QPointF>(variant).x()),
                                                                                               QString::number(qvariant_cast<QPointF>(variant).y()));
    else if (variant.canConvert(QVariant::RectF)) return QString("(%1,%2,%3,%4)").arg(QString::number(qvariant_cast<QRectF>(variant).x()),
                                                                                                         QString::number(qvariant_cast<QRectF>(variant).y()),
                                                                                                         QString::number(qvariant_cast<QRectF>(variant).width()),
                                                                                                         QString::number(qvariant_cast<QRectF>(variant).height()));
    return QString();
}

void File::fromString(const QString &key, const QString &value)
{
    if (value[0] == '[') /* QVariantList */ {
        QStringList values = value.mid(1, value.size()-2).split(", ");
        foreach(const QString &word, values) fromString(key, word);
    }
    else if (value[0] == '(') {
        QStringList values = value.split(',');
        if (values.size() == 2) /* QPointF */ {
            values[1].chop(1);
            QPointF point(values[0].mid(1).toFloat(), values[1].toFloat());
            if (key != "Points") set(key, point);
            else appendPoint(point);
        }
        else /* QRectF */ {
            values[3].chop(1);
            QRectF rect(values[0].mid(1).toFloat(), values[1].toFloat(), values[2].toFloat(), values[3].toFloat());
            if (key != "Rects") set(key, rect);
            else appendRect(rect);
        }
    }
    else set(key, value);
}

/* File - global methods */
QDebug br::operator<<(QDebug dbg, const File &file)
{
    return dbg.nospace() << qPrintable(file.flat());
}

QDataStream &br::operator<<(QDataStream &stream, const File &file)
{
    return stream << file.name << file.m_metadata;
}

QDataStream &br::operator>>(QDataStream &stream, File &file)
{
    return stream >> file.name >> file.m_metadata;
    const QVariant label = file.m_metadata.value("Label");
    if (!label.isNull()) file.set("Label", label); // Trigger population of Globals->classes
}

/* FileList - public methods */
FileList::FileList(const QStringList &files)
{
    reserve(files.size());
    foreach (const QString &file, files)
        append(file);
}

FileList::FileList(int n)
{
    reserve(n);
    for (int i=0; i<n; i++)
        append(File());
}

QStringList FileList::flat() const
{
    QStringList flat; flat.reserve(size());
    foreach (const File &file, *this) flat.append(file.flat());
    return flat;
}

QStringList FileList::names() const
{
    QStringList names;
    foreach (const File &file, *this)
        names.append(file);
    return names;
}

void FileList::sort(const QString& key)
{
    if (size() <= 1) return;

    QList<QString> metadata;
    FileList sortedList;

    for (int i = 0; i < size(); i++) {
        if (at(i).contains(key))
            metadata.append(at(i).get<QString>(key));
        else sortedList.push_back(at(i));
    }

    typedef QPair<QString,int> Pair;
    foreach (const Pair &pair, Common::Sort(metadata, true))
        sortedList.prepend(at(pair.second));

    *this = sortedList;
}

QList<float> FileList::labels() const
{
    QList<float> labels; labels.reserve(size());
    foreach (const File &f, *this)
        labels.append(f.label());
    return labels;
}

QList<int> FileList::crossValidationPartitions() const
{
    QList<int> crossValidationPartitions; crossValidationPartitions.reserve(size());
    foreach (const File &f, *this)
        crossValidationPartitions.append(f.get<int>("Cross_Validation_Partition", 0));
    return crossValidationPartitions;
}

int FileList::failures() const
{
    int failures = 0;
    foreach (const File &file, *this)
        if (file.get<bool>("FTO", false) || file.get<bool>("FTE", false))
            failures++;
    return failures;
}

/* Template - global methods */
QDataStream &br::operator<<(QDataStream &stream, const Template &t)
{
    return stream << static_cast<const QList<cv::Mat>&>(t) << t.file;
}

QDataStream &br::operator>>(QDataStream &stream, Template &t)
{
    return stream >> static_cast<QList<cv::Mat>&>(t) >> t.file;
}

/* TemplateList - public methods */
TemplateList TemplateList::fromGallery(const br::File &gallery)
{
    TemplateList templates;
    foreach (const br::File &file, gallery.split()) {
        QScopedPointer<Gallery> i(Gallery::make(file));
        TemplateList newTemplates = i->read();
        newTemplates = newTemplates.mid(gallery.get<int>("pos", 0), gallery.get<int>("length", -1));

        const int step = gallery.get<int>("step", 1);
        if (step > 1) {
            TemplateList downsampled; downsampled.reserve(newTemplates.size()/step);
            for (int i=0; i<newTemplates.size(); i+=step)
                downsampled.append(newTemplates[i]);
            newTemplates = downsampled;
        }

        if (gallery.get<bool>("reduce", false)) newTemplates = newTemplates.reduced();
        const int crossValidate = gallery.get<int>("crossValidate");
        if (crossValidate > 0) srand(0);

        // If file is a Format not a Gallery
        if (newTemplates.isEmpty())
            newTemplates.append(file);

        // Propogate metadata
        for (int i=0; i<newTemplates.size(); i++) {
            newTemplates[i].file.append(gallery.localMetadata());
            newTemplates[i].file.append(file.localMetadata());
            newTemplates[i].file.set("Index", i+templates.size());
            if (crossValidate > 0) newTemplates[i].file.set("Cross_Validation_Partition", rand()%crossValidate);
        }

        if (!templates.isEmpty() && gallery.get<bool>("merge", false)) {
            if (newTemplates.size() != templates.size())
                qFatal("Inputs must be the same size in order to merge.");
            for (int i=0; i<templates.size(); i++)
                templates[i].merge(newTemplates[i]);
        } else {
            templates += newTemplates;
        }
    }

    return templates;
}

TemplateList TemplateList::relabel(const TemplateList &tl)
{
    QHash<int,int> labels;
    foreach (int label, tl.labels<int>())
        if (!labels.contains(label))
            labels.insert(label, labels.size());

    TemplateList result = tl;
    for (int i=0; i<result.size(); i++)
        result[i].file.setLabel(labels[result[i].file.label()]);
    return result;
}

/* Object - public methods */
QStringList Object::parameters() const
{
    QStringList parameters;

    for (int i = firstAvailablePropertyIdx; i < metaObject()->propertyCount();i++) {
        QMetaProperty property = metaObject()->property(i);
        if (property.isStored(this)) continue;
        parameters.append(QString("%1 %2 = %3").arg(property.typeName(), property.name(), property.read(this).toString()));
    }
    return parameters;
}

QStringList Object::arguments() const
{
    QStringList arguments;
    for (int i=metaObject()->propertyOffset(); i<metaObject()->propertyCount(); i++) {
        QMetaProperty property = metaObject()->property(i);
        if (property.isStored(this)) continue;
        arguments.append(argument(i));
    }
    return arguments;
}

QString Object::argument(int index) const
{
    if ((index < 0) || (index > metaObject()->propertyCount())) return "";
    const QMetaProperty property = metaObject()->property(index);
    const QVariant variant = property.read(this);
    const QString type = property.typeName();

    if (type.startsWith("QList<") && type.endsWith(">")) {
        QStringList strings;

        if (type == "QList<float>") {
            foreach (float value, variant.value< QList<float> >())
                strings.append(QString::number(value));
        } else if (type == "QList<int>") {
            foreach (int value, variant.value< QList<int> >())
                strings.append(QString::number(value));
        } else if (type == "QList<br::Transform*>") {
            foreach (Transform *transform, variant.value< QList<Transform*> >())
                strings.append(transform->description());
        } else if (type == "QList<br::Distance*>") {
            foreach (Distance *distance, variant.value< QList<Distance*> >())
                strings.append(distance->description());
        } else {
            qFatal("Unrecognized type: %s", qPrintable(type));
        }

        return "[" + strings.join(",") + "]";
    } else if (type == "br::Transform*") {
        return variant.value<Transform*>()->description();
    } else if (type == "br::Distance*") {
        return variant.value<Distance*>()->description();
    } else if (type == "QStringList") {
        return "[" + variant.toStringList().join(",") + "]";
    }

    return variant.toString();
}

QString Object::description() const
{
    QString argumentString = arguments().join(",");
    return objectName() + (argumentString.isEmpty() ? "" : ("(" + argumentString + ")"));
}

void Object::store(QDataStream &stream) const
{
    // Start from 1 to skip QObject::objectName
    for (int i=1; i<metaObject()->propertyCount(); i++) {
        QMetaProperty property = metaObject()->property(i);
        if (!property.isStored(this))
            continue;

        const QString type = property.typeName();
        if (type == "QList<br::Transform*>") {
            foreach (Transform *transform, property.read(this).value< QList<Transform*> >())
                transform->store(stream);
        } else if (type == "QList<br::Distance*>") {
            foreach (Distance *distance, property.read(this).value< QList<Distance*> >())
                distance->store(stream);
        } else if (type == "br::Transform*") {
            property.read(this).value<Transform*>()->store(stream);
        } else if (type == "br::Distance*") {
            property.read(this).value<Distance*>()->store(stream);
        } else if (type == "bool") {
            stream << property.read(this).toBool();
        } else if (type == "int") {
            stream << property.read(this).toInt();
        } else if (type == "float") {
            stream << property.read(this).toFloat();
        } else if (type == "double") {
            stream << property.read(this).toDouble();
        } else if (type == "QString") {
            stream << property.read(this).toString();
        } else if (type == "QStringList") {
            stream << property.read(this).toStringList();
        } else {
            qFatal("Can't serialize value of type: %s", qPrintable(type));
        }
    }
}

void Object::load(QDataStream &stream)
{
    // Start from 1 to skip QObject::objectName
    for (int i=1; i<metaObject()->propertyCount(); i++) {
        QMetaProperty property = metaObject()->property(i);
        if (!property.isStored(this))
            continue;

        const QString type = property.typeName();
        if (type == "QList<br::Transform*>") {
            foreach (Transform *transform, property.read(this).value< QList<Transform*> >())
                transform->load(stream);
        } else if (type == "QList<br::Distance*>") {
            foreach (Distance *distance, property.read(this).value< QList<Distance*> >())
                distance->load(stream);
        } else if (type == "br::Transform*") {
            property.read(this).value<Transform*>()->load(stream);
        } else if (type == "br::Distance*") {
            property.read(this).value<Distance*>()->load(stream);
        } else if (type == "bool") {
            bool value;
            stream >> value;
            property.write(this, value);
        } else if (type == "int") {
            int value;
            stream >> value;
            property.write(this, value);
        } else if (type == "float") {
            float value;
            stream >> value;
            property.write(this, value);
        } else if (type == "double") {
            double value;
            stream >> value;
            property.write(this, value);
        } else if (type == "QString") {
            QString value;
            stream >> value;
            property.write(this, value);
        } else if (type == "QStringList") {
            QStringList value;
            stream >> value;
            property.write(this, value);
        } else {
            qFatal("Can't serialize value of type: %s", qPrintable(type));
        }
    }

    init();
}

void Object::setProperty(const QString &name, const QString &value)
{
    QString type;
    int index = metaObject()->indexOfProperty(qPrintable(name));
    if (index != -1) type = metaObject()->property(index).typeName();
    else             type = "";

    QVariant variant;
    if (type.startsWith("QList<") && type.endsWith(">")) {
        if (!value.startsWith('[')) qFatal("Expected a list.");
        const QStringList strings = parse(value.mid(1, value.size()-2));

        if (type == "QList<float>") {
            QList<float> values;
            foreach (const QString &string, strings)
                values.append(string.toFloat());
            variant.setValue(values);
        } else if (type == "QList<int>") {
            QList<int> values;
            foreach (const QString &string, strings)
                values.append(string.toInt());
            variant.setValue(values);
        } else if (type == "QList<br::Transform*>") {
            QList<Transform*> values;
            foreach (const QString &string, strings)
                values.append(Transform::make(string, this));
            variant.setValue(values);
        } else if (type == "QList<br::Distance*>") {
            QList<Distance*> values;
            foreach (const QString &string, strings)
                values.append(Distance::make(string, this));
            variant.setValue(values);
        } else {
            qFatal("Unrecognized type: %s", qPrintable(type));
        }
    } else if (type == "br::Transform*") {
        variant.setValue(Transform::make(value, this));
    } else if (type == "br::Distance*") {
        variant.setValue(Distance::make(value, this));
    } else if (type == "QStringList") {
        variant.setValue(parse(value.mid(1, value.size()-2)));
    } else if (type == "bool") {
        if      (value.isEmpty())  variant = true;
        else if (value == "false") variant = false;
        else if (value == "true")  variant = true;
        else                       variant = value;
    } else {
        variant = value;
    }

    if (!QObject::setProperty(qPrintable(name), variant) && !type.isEmpty())
        qFatal("Failed to set %s::%s to: %s %s",
                metaObject()->className(), qPrintable(name), qPrintable(value), qPrintable(type));
}

QStringList br::Object::parse(const QString &string, char split)
{
    return QtUtils::parse(string, split);
}

/* Object - private methods */
void Object::init(const File &file_)
{
    this->file = file_;

    // Set name
    QString name = metaObject()->className();
    if (name.startsWith("br::")) name = name.right(name.size()-4);

    firstAvailablePropertyIdx = metaObject()->propertyCount();

    const QMetaObject * baseClass = metaObject();
    const QMetaObject * superClass = metaObject()->superClass();

    while (superClass != NULL) {
        const QMetaObject * nextClass = superClass->superClass();

        // baseClass <- something <- br::Object
        // baseClass is the highest class whose properties we can set via positional arguments
        if (nextClass && !strcmp(nextClass->className(),"br::Object")) {
            firstAvailablePropertyIdx = baseClass->propertyOffset();
        }

        QString superClassName = superClass->className();

        // strip br:: prefix from superclass name
        if (superClassName.startsWith("br::"))
            superClassName = superClassName.right(superClassName.size()-4);

        // Strip superclass name from base class name (e.g. PipeTransform -> Pipe)
        if (name.endsWith(superClassName))
            name = name.left(name.size() - superClassName.size());
        baseClass = superClass;
        superClass = superClass->superClass();

    }
    setObjectName(name);

    // Reset all properties
    for (int i=0; i<metaObject()->propertyCount(); i++) {
        QMetaProperty property = metaObject()->property(i);
        if (property.isResettable())
            if (!property.reset(this))
                qFatal("Failed to reset %s::%s", metaObject()->className(), property.name());
    }

    foreach (QString key, file.localKeys()) {
        const QString value = file.value(key).toString();

        if (key.startsWith(("_Arg"))) {
            int argument_number =  key.mid(4).toInt();
            int target_idx = argument_number + firstAvailablePropertyIdx;

            if (target_idx >= metaObject()->propertyCount()) {
                qWarning("too many arguments for transform, ignoring %s\n", qPrintable(value));
                continue;
            }
            key = metaObject()->property(target_idx).name();
        }
        setProperty(key, value);
    }

    init();
}

/* Context - public methods */
int br::Context::blocks(int size) const
{
    return std::ceil(1.f*size/blockSize);
}

bool br::Context::contains(const QString &name)
{
    QByteArray bytes = name.toLocal8Bit();
    const char * c_name = bytes.constData();

    for (int i=0; i<metaObject()->propertyCount(); i++)
        if (!strcmp(c_name, metaObject()->property(i).name()))
            return true;
    return false;
}

void br::Context::printStatus()
{
    if (verbose || quiet || (totalSteps < 2)) return;
    const float p = progress();
    if (p < 1) {
        int s = timeRemaining();
        int h = s / (60*60);
        int m = (s - h*60*60) / 60;
        s = (s - h*60*60 - m*60);
        fprintf(stderr, "%05.2f%%  REMAINING=%02d:%02d:%02d  COUNT=%g  \r", 100 * p, h, m, s, totalSteps);
    }
}

float br::Context::progress() const
{
    if (totalSteps == 0) return -1;
    return currentStep / totalSteps;
}

void br::Context::setProperty(const QString &key, const QString &value)
{
    Object::setProperty(key, value);
    qDebug("Set %s%s", qPrintable(key), value.isEmpty() ? "" : qPrintable(" to " + value));

    if (key == "parallelism") {
        const int maxThreads = std::max(1, QThread::idealThreadCount());
        QThreadPool::globalInstance()->setMaxThreadCount(parallelism ? std::min(maxThreads, abs(parallelism)) : maxThreads);
    } else if (key == "log") {
        logFile.close();
        if (log.isEmpty()) return;

        logFile.setFileName(log);
        QtUtils::touchDir(logFile);
        logFile.open(QFile::Append);
        logFile.write("================================================================================\n");
    }
}

int br::Context::timeRemaining() const
{
    const float p = progress();
    if (p < 0) return -1;
    return std::max(0, int((1 - p) / p * startTime.elapsed())) / 1000;
}

bool br::Context::checkSDKPath(const QString &sdkPath)
{
    return QFileInfo(sdkPath + "/share/openbr/openbr.bib").exists();
}

void br::Context::initialize(int &argc, char *argv[], const QString &sdkPath)
{
    // We take in argc as a reference due to:
    //   https://bugreports.qt-project.org/browse/QTBUG-5637
    // QApplication should be initialized before anything else.
    // Since we can't ensure that it gets deleted last, we never delete it.
    static QCoreApplication *application = NULL;
    if (application == NULL) {
#ifndef BR_EMBEDDED
        application = new QApplication(argc, argv);
#else
        application = new QCoreApplication(argc, argv);
#endif
    }

    if (Globals == NULL) {
        Globals = new Context();
        Globals->init(File());
    }

    initializeQt(sdkPath);

#ifdef BR_DISTRIBUTED
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Cobr_rank(MPI_CObr_WORLD, &rank);
    MPI_Cobr_size(MPI_CObr_WORLD, &size);
    if (!quiet) qDebug() << "OpenBR distributed process" << rank << "of" << size;
#endif // BR_DISTRIBUTED
}

void br::Context::initializeQt(QString sdkPath)
{
    if (Globals == NULL) {
        Globals = new Context();
        Globals->init(File());
    }

    QCoreApplication::setOrganizationName(COMPANY_NAME);
    QCoreApplication::setApplicationName(PRODUCT_NAME);
    QCoreApplication::setApplicationVersion(PRODUCT_VERSION);

    qRegisterMetaType< QList<float> >();
    qRegisterMetaType< QList<int> >();
    qRegisterMetaType< br::Transform* >();
    qRegisterMetaType< QList<br::Transform*> >();
    qRegisterMetaType< br::Distance* >();
    qRegisterMetaType< QList<br::Distance*> >();
    qRegisterMetaType< cv::Mat >();

    qInstallMessageHandler(messageHandler);

    // Search for SDK
    if (sdkPath.isEmpty()) {
        QStringList checkPaths; checkPaths << QDir::currentPath() << QCoreApplication::applicationDirPath();

        bool foundSDK = false;
        foreach (const QString &path, checkPaths) {
            if (foundSDK) break;
            QDir dir(path);
            do {
                sdkPath = dir.absolutePath();
                foundSDK = checkSDKPath(sdkPath);
                dir.cdUp();
            } while (!foundSDK && !dir.isRoot());
        }

        if (!foundSDK) qFatal("Unable to locate SDK automatically.");
    } else {
        if (!checkSDKPath(sdkPath)) qFatal("Unable to locate SDK from %s.", qPrintable(sdkPath));
    }

    Globals->sdkPath = sdkPath;

    // Trigger registered initializers
    QList< QSharedPointer<Initializer> > initializers = Factory<Initializer>::makeAll();
    foreach (const QSharedPointer<Initializer> &initializer, initializers)
        initializer->initialize();
}

void br::Context::finalize()
{
    // Is anyone still running?
    QThreadPool::globalInstance()->waitForDone();

    // Trigger registered finalizers
    QList< QSharedPointer<Initializer> > initializers = Factory<Initializer>::makeAll();
    foreach (const QSharedPointer<Initializer> &initializer, initializers)
        initializer->finalize();

#ifdef BR_DISTRIBUTED
    MPI_Finalize();
#endif // BR_DISTRIBUTED

    delete Globals;
    Globals = NULL;
}

QString br::Context::about()
{
    return QString("%1 %2 %3").arg(PRODUCT_NAME, PRODUCT_VERSION, LEGAL_COPYRIGHT);
}

QString br::Context::version()
{
    return PRODUCT_VERSION;
}

QString br::Context::scratchPath()
{
    return QString("%1/%2-%3.%4").arg(QDir::homePath(), PRODUCT_NAME, QString::number(PRODUCT_VERSION_MAJOR), QString::number(PRODUCT_VERSION_MINOR));
}

void br::Context::messageHandler(QtMsgType type, const QMessageLogContext &context, const QString &msg)
{
    // Something about this method is not thread safe, and will lead to crashes if qDebug
    // statements are called from multiple threads. Unless we lock the whole thing...
    static QMutex generalLock;
    QMutexLocker locker(&generalLock);

    QString txt;
    switch (type) {
      case QtDebugMsg:
        if (Globals->quiet) return;
        txt = QString("%1\n").arg(msg);
        break;
      case QtWarningMsg:
        txt = QString("Warning: %1\n").arg(msg);
        break;
      case QtCriticalMsg:
        txt = QString("Critical: %1\n").arg(msg);
        break;
      case QtFatalMsg:
        txt = QString("Fatal: %1\n").arg(msg);
        break;
    }

    std::cerr << txt.toStdString();
    Globals->mostRecentMessage = txt;

    if (Globals->logFile.isWritable()) {
        Globals->logFile.write(qPrintable(txt));
        Globals->logFile.flush();
    }

    if (type == QtFatalMsg) {
        // Write debug output then close
        qDebug("  File: %s\n  Function: %s\n  Line: %d", qPrintable(context.file), qPrintable(context.function), context.line);
        Globals->finalize();
        //QCoreApplication::exit(-1);
        abort();
    }
}

Context *br::Globals = NULL;

/* Output - public methods */
void Output::setBlock(int rowBlock, int columnBlock)
{
    offset = QPoint((columnBlock == -1) ? 0 : Globals->blockSize*columnBlock,
                    (rowBlock == -1) ? 0 : Globals->blockSize*rowBlock);
    if (!next.isNull()) next->setBlock(rowBlock, columnBlock);
}

void Output::setRelative(float value, int i, int j)
{
    set(value, i+offset.y(), j+offset.x());
    if (!next.isNull()) next->setRelative(value, i, j);
}

Output *Output::make(const File &file, const FileList &targetFiles, const FileList &queryFiles)
{
    Output *output = NULL;
    FileList files = file.split();
    if (files.isEmpty()) files.append(File());
    foreach (const File &subfile, files) {
        Output *newOutput = Factory<Output>::make(subfile);
        newOutput->initialize(targetFiles, queryFiles);
        newOutput->next = QSharedPointer<Output>(output);
        output = newOutput;
    }
    return output;
}

void Output::reformat(const FileList &targetFiles, const FileList &queryFiles, const File &simmat, const File &output)
{
    qDebug("Reformating %s to %s", qPrintable(simmat.flat()), qPrintable(output.flat()));

    Mat m = BEE::readSimmat(simmat);

    QSharedPointer<Output> o(Factory<Output>::make(output));
    o->initialize(targetFiles, queryFiles);

    const int rows = queryFiles.size();
    const int columns = targetFiles.size();
    for (int i=0; i<rows; i++)
        for (int j=0; j<columns; j++)
            o->setRelative(m.at<float>(i,i), i, j);
}

/* Output - protected methods */
void Output::initialize(const FileList &targetFiles, const FileList &queryFiles)
{
    this->targetFiles = targetFiles;
    this->queryFiles = queryFiles;
    selfSimilar = (queryFiles == targetFiles) && (targetFiles.size() > 1) && (queryFiles.size() > 1);
}

/* MatrixOutput - public methods */
void MatrixOutput::initialize(const FileList &targetFiles, const FileList &queryFiles)
{
    Output::initialize(targetFiles, queryFiles);
    data.create(queryFiles.size(), targetFiles.size(), CV_32FC1);
}

MatrixOutput *MatrixOutput::make(const FileList &targetFiles, const FileList &queryFiles)
{
    return dynamic_cast<MatrixOutput*>(Output::make(".Matrix", targetFiles, queryFiles));
}

/* MatrixOutput - protected methods */
QString MatrixOutput::toString(int row, int column) const
{
    if (targetFiles[column] == "Label")
        return File::subject(data.at<float>(row,column));
    return QString::number(data.at<float>(row,column));
}

/* MatrixOutput - private methods */
void MatrixOutput::set(float value, int i, int j)
{
    data.at<float>(i,j) = value;
}

BR_REGISTER(Output, MatrixOutput)

/* Gallery - public methods */
TemplateList Gallery::read()
{
    TemplateList templates;
    bool done = false;
    while (!done) templates.append(readBlock(&done));
    return templates;
}

FileList Gallery::files()
{
    FileList files;
    bool done = false;
    while (!done) files.append(readBlock(&done).files());
    return files;
}

void Gallery::writeBlock(const TemplateList &templates)
{
    foreach (const Template &t, templates) write(t);
    if (!next.isNull()) next->writeBlock(templates);
}

Gallery *Gallery::make(const File &file)
{
    Gallery *gallery = NULL;
    foreach (const File &f, file.split()) {
        Gallery *next = gallery;
        gallery = Factory<Gallery>::make(f);
        gallery->next = QSharedPointer<Gallery>(next);
    }
    return gallery;
}

static TemplateList Downsample(const TemplateList &templates, const Transform *transform)
{
    // Return early when no downsampling is required
    if ((transform->classes == std::numeric_limits<int>::max()) &&
        (transform->instances == std::numeric_limits<int>::max()) &&
        (transform->fraction >= 1))
        return templates;

    const bool atLeast = transform->instances < 0;
    const int instances = abs(transform->instances);

    QList<int> allLabels = templates.labels<int>();
    QList<int> uniqueLabels = allLabels.toSet().toList();
    qSort(uniqueLabels);

    QMap<int,int> counts = templates.labelCounts(instances != std::numeric_limits<int>::max());
    if ((instances != std::numeric_limits<int>::max()) && (transform->classes != std::numeric_limits<int>::max()))
        foreach (int label, counts.keys())
            if (counts[label] < instances)
                counts.remove(label);
    uniqueLabels = counts.keys();
    if ((transform->classes != std::numeric_limits<int>::max()) && (uniqueLabels.size() < transform->classes))
        qWarning("Downsample requested %d classes but only %d are available.", transform->classes, uniqueLabels.size());

    Common::seedRNG();
    QList<int> selectedLabels = uniqueLabels;
    if (transform->classes < uniqueLabels.size()) {
        std::random_shuffle(selectedLabels.begin(), selectedLabels.end());
        selectedLabels = selectedLabels.mid(0, transform->classes);
    }

    TemplateList downsample;
    for (int i=0; i<selectedLabels.size(); i++) {
        const int selectedLabel = selectedLabels[i];
        QList<int> indices;
        for (int j=0; j<allLabels.size(); j++)
            if ((allLabels[j] == selectedLabel) && (!templates.value(j).file.get<bool>("FTE", false)))
                indices.append(j);

        std::random_shuffle(indices.begin(), indices.end());
        const int max = atLeast ? indices.size() : std::min(indices.size(), instances);
        for (int j=0; j<max; j++)
            downsample.append(templates.value(indices[j]));
    }

    if (transform->fraction < 1) {
        std::random_shuffle(downsample.begin(), downsample.end());
        downsample = downsample.mid(0, downsample.size()*transform->fraction);
    }

    return downsample;
}

/*!
 * \ingroup transforms
 * \brief Clones the transform so that it can be applied independently.
 *
 * \em Independent transforms expect single-matrix templates.
 */
class Independent : public MetaTransform
{
    Q_PROPERTY(QList<Transform*> transforms READ get_transforms WRITE set_transforms STORED false)
    BR_PROPERTY(QList<Transform*>, transforms, QList<Transform*>())

public:
    /*!
     * \brief Independent
     * \param transform
     */
    Independent(Transform *transform)
    {
        transform->setParent(this);
        transforms.append(transform);
        file = transform->file;
        trainable = transform->trainable;
        setObjectName(transforms.first()->objectName());
    }

private:
    Transform *clone() const
    {
        return new Independent(transforms.first()->clone());
    }

    static void _train(Transform *transform, const TemplateList *data)
    {
        transform->train(*data);
    }

    void train(const TemplateList &data)
    {
        // Don't bother if the transform is untrainable
        if (!trainable) return;

        QList<TemplateList> templatesList;
        foreach (const Template &t, data) {
            if ((templatesList.size() != t.size()) && !templatesList.isEmpty())
                qWarning("Independent::train template %s of size %d differs from expected size %d.", qPrintable(t.file.name), t.size(), templatesList.size());
            while (templatesList.size() < t.size())
                templatesList.append(TemplateList());
            for (int i=0; i<t.size(); i++)
                templatesList[i].append(Template(t.file, t[i]));
        }

        while (transforms.size() < templatesList.size())
            transforms.append(transforms.first()->clone());

        for (int i=0; i<templatesList.size(); i++)
            templatesList[i] = Downsample(templatesList[i], transforms[i]);

        QFutureSynchronizer<void> futures;
        for (int i=0; i<templatesList.size(); i++) {
            if (Globals->parallelism) futures.addFuture(QtConcurrent::run(_train, transforms[i], &templatesList[i]));
            else                                                          _train (transforms[i], &templatesList[i]);
        }
        QtUtils::releaseAndWait(futures);
    }

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;
        QList<Mat> mats;
        for (int i=0; i<src.size(); i++) {
            transforms[i%transforms.size()]->project(Template(src.file, src[i]), dst);
            mats.append(dst);
            dst.clear();
        }
        dst.append(mats);
    }

    void store(QDataStream &stream) const
    {
        const int size = transforms.size();
        stream << size;
        for (int i=0; i<size; i++)
            transforms[i]->store(stream);
    }

    void load(QDataStream &stream)
    {
        int size;
        stream >> size;
        while (transforms.size() < size)
            transforms.append(transforms.first()->clone());
        for (int i=0; i<size; i++)
            transforms[i]->load(stream);
    }
};

/* Transform - public methods */
Transform::Transform(bool _independent, bool _trainable)
{
    independent = _independent;
    trainable = _trainable;
    classes = std::numeric_limits<int>::max();
    instances = std::numeric_limits<int>::max();
    fraction = 1;
}

Transform *Transform::make(QString str, QObject *parent)
{
    // Check for custom transforms
    if (Globals->abbreviations.contains(str))
        return make(Globals->abbreviations[str], parent);

    { // Check for use of '!' as shorthand for Expand
        str.replace("!","+Expand+");
    }

    { // Check for use of '+' as shorthand for Pipe(...)
        QStringList words = parse(str, '+');
        if (words.size() > 1)
            return make("Pipe([" + words.join(",") + "])", parent);
    }

    { // Check for use of '/' as shorthand for Fork(...)
        QStringList words = parse(str, '/');
        if (words.size() > 1)
            return make("Fork([" + words.join(",") + "])", parent);
    }

    // Check for use of '{...}' as shorthand for Cache(...)
    if (str.startsWith('{') && str.endsWith('}'))
        return make("Cache(" + str.mid(1, str.size()-2) + ")", parent);

    // Check for use of '<...>' as shorthand for LoadStore(...)
    if (str.startsWith('<') && str.endsWith('>'))
        return make("LoadStore(" + str.mid(1, str.size()-2) + ")", parent);

    // Check for use of '(...)' to change order of operations
    if (str.startsWith('(') && str.endsWith(')'))
        return make(str.mid(1, str.size()-2), parent);

    File f = "." + str;
    Transform *transform = Factory<Transform>::make(f);

    if (transform->independent)
        transform = new Independent(transform);
    transform->setParent(parent);
    return transform;
}

Transform *Transform::clone() const
{
    Transform *clone = Factory<Transform>::make(file.flat());
    clone->classes = classes;
    clone->instances = instances;
    clone->fraction = fraction;
    return clone;
}

static void _project(const Transform *transform, const Template *src, Template *dst)
{
    try {
        transform->project(*src, *dst);
    } catch (...) {
        qWarning("Exception triggered when processing %s with transform %s", qPrintable(src->file.flat()), qPrintable(transform->objectName()));
        *dst = Template(src->file);
        dst->file.set("FTE", true);
    }
}

// Default project(TemplateList) calls project(Template) separately for each element
void Transform::project(const TemplateList &src, TemplateList &dst) const
{
    dst.reserve(src.size());

    // There are certain conditions where we should process the templates in serial,
    // but generally we'd prefer to process them in parallel.
    if ((src.size() < 2) ||
        (QThreadPool::globalInstance()->activeThreadCount() >= QThreadPool::globalInstance()->maxThreadCount()) ||
        (Globals->parallelism == 0)) {

        foreach (const Template &t, src) {
            dst.append(Template());
            _project(this, &t, &dst.last());
        }
    } else {
        for (int i=0; i<src.size(); i++)
            dst.append(Template());
        QFutureSynchronizer<void> futures;
        for (int i=0; i<dst.size(); i++)
            futures.addFuture(QtConcurrent::run(_project, this, &src[i], &dst[i]));
        QtUtils::releaseAndWait(futures);
    }
}

static void _backProject(const Transform *transform, const Template *dst, Template *src)
{
    try {
        transform->backProject(*dst, *src);
    } catch (...) {
        qWarning("Exception triggered when processing %s with transform %s", qPrintable(src->file.flat()), qPrintable(transform->objectName()));
        *src = Template(dst->file);
        src->file.set("FTE", true);
    }
}

void Transform::backProject(const TemplateList &dst, TemplateList &src) const
{
    src.reserve(dst.size());
    for (int i=0; i<dst.size(); i++) src.append(Template());

    QFutureSynchronizer<void> futures;
    for (int i=0; i<dst.size(); i++)
        if (Globals->parallelism) futures.addFuture(QtConcurrent::run(_backProject, this, &dst[i], &src[i]));
        else                                                          _backProject (this, &dst[i], &src[i]);
    QtUtils::releaseAndWait(futures);
}

/* Distance - public methods */
Distance *Distance::make(QString str, QObject *parent)
{
    // Check for custom transforms
    if (Globals->abbreviations.contains(str))
        return make(Globals->abbreviations[str], parent);

    { // Check for use of '+' as shorthand for Pipe(...)
        QStringList words = parse(str, '+');
        if (words.size() > 1)
            return make("Pipe([" + words.join(",") + "])", parent);
    }

    File f = "." + str;
    Distance *distance = Factory<Distance>::make(f);

    distance->setParent(parent);
    return distance;
}

void Distance::compare(const TemplateList &target, const TemplateList &query, Output *output) const
{
    const bool stepTarget = target.size() > query.size();
    const int totalSize = std::max(target.size(), query.size());
    int stepSize = ceil(float(totalSize) / float(std::max(1, abs(Globals->parallelism))));
    QFutureSynchronizer<void> futures;
    for (int i=0; i<totalSize; i+=stepSize) {
        const TemplateList &targets(stepTarget ? TemplateList(target.mid(i, stepSize)) : target);
        const TemplateList &queries(stepTarget ? query : TemplateList(query.mid(i, stepSize)));
        const int targetOffset = stepTarget ? i : 0;
        const int queryOffset = stepTarget ? 0 : i;
        if (Globals->parallelism) futures.addFuture(QtConcurrent::run(this, &Distance::compareBlock, targets, queries, output, targetOffset, queryOffset));
        else                                                                           compareBlock (targets, queries, output, targetOffset, queryOffset);
    }
    QtUtils::releaseAndWait(futures);
}

QList<float> Distance::compare(const TemplateList &targets, const Template &query) const
{
    QList<float> scores; scores.reserve(targets.size());
    foreach (const Template &target, targets)
        scores.append(compare(target, query));
    return scores;
}

/* Distance - private methods */
void Distance::compareBlock(const TemplateList &target, const TemplateList &query, Output *output, int targetOffset, int queryOffset) const
{
    for (int i=0; i<query.size(); i++)
        for (int j=0; j<target.size(); j++)
            output->setRelative(compare(target[j], query[i]), i+queryOffset, j+targetOffset);
}
