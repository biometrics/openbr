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

#include <QCoreApplication>
#include <QCryptographicHash>
#include <QFutureSynchronizer>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMetaProperty>
#include <qnumeric.h>
#include <QPointF>
#include <QProcess>
#include <QRect>
#include <QRegExp>
#include <QThreadPool>
#include <QtConcurrentRun>
#include <algorithm>
#include <iostream>

#ifndef BR_EMBEDDED
#include <QApplication>
#endif

#include "openbr_plugin.h"
#include "version.h"
#include "core/bee.h"
#include "core/common.h"
#include "core/opencvutils.h"
#include "core/qtutils.h"
#include "openbr/plugins/openbr_internal.h"

using namespace br;
using namespace cv;

static const QMetaObject *getInterface(const QObject *obj)
{
    const QMetaObject *baseClass = obj->metaObject();

    while (strcmp(baseClass->superClass()->className(), "br::Object") != 0)
        baseClass = baseClass->superClass();

    return baseClass;
}

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
        else values.append(key + "=" + QtUtils::toString(value));
    }

    QString flat = name;
    if (!values.isEmpty()) flat += "[" + values.join(", ") + "]";
    return flat;
}

QString File::hash() const
{
    return QtUtils::shortTextHash(flat());
}

void File::append(const QVariantMap &metadata)
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
            if (!contains("separator")) set("separator", QString(";"));
            name += value("separator").toString() + other.name;
        }
    }
    append(other.m_metadata);
    fte = fte | other.fte;
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
        // If file metadata is empty after this constructor, it means that this is the
        // file corresponding to *this.m_metadata, so we append its metadata to get
        // the correct functionality
        if (file.m_metadata.isEmpty())
            file.append(m_metadata);
        files.append(file);
    }
    return files;
}

QString File::resolved() const
{
    if (exists()) return name;
    QStringList paths = get<QString>("path").split(";", QString::SkipEmptyParts);
    foreach (const QString &path, paths) {
        const File resolved = path + "/" + name;
        if (resolved.exists()) return resolved;
    }
    foreach (const QString &path, paths) {
        const File resolved = path + "/" + fileName();
        if (resolved.exists()) return resolved;
    }
    return name;
}

bool File::contains(const QString &key) const
{
    return m_metadata.contains(key) || Globals->contains(key) || key == "name";
}

bool File::contains(const QStringList &keys) const
{
    foreach (const QString &key, keys)
        if (!contains(key))
            return false;
    return true;
}

QVariant File::value(const QString &key) const
{
    return m_metadata.contains(key) ? m_metadata.value(key) : (key == "name" ? name : Globals->property(qPrintable(key)));
}

QVariant File::parse(const QString &value)
{
    return QtUtils::fromString(value);
}

void File::set(const QString &key, const QString &value)
{
    set(key, QtUtils::fromString(value));
}

bool File::getBool(const QString &key, bool defaultValue) const
{
    if (!contains(key)) return defaultValue;
    QVariant variant = value(key);
    if (variant.isNull() || !variant.canConvert<bool>()) return true;
    return variant.value<bool>();
}

QList<QPointF> File::namedPoints() const
{
    QList<QPointF> landmarks;
    foreach (const QString &key, localMetadata().keys()) {
        const QVariant &variant = m_metadata[key];
        if (variant.canConvert<QPointF>()) {
            const QPointF point = variant.value<QPointF>();
            if (!qIsNaN(point.x()) && !qIsNaN(point.y()))
                landmarks.append(point);
        }
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
        else if (variant.canConvert<QList<QRectF> >()) {
            QList<QRectF> list = variant.value<QList<QRectF> >();
            for (int i=0;i < list.size();i++)
            {
                rects.append(list[i]);
            }
        }
    }
    return rects;
}

QList<QRectF> File::rects() const
{
    QList<QRectF> rects;
    foreach (const QVariant &rect, m_metadata["Rects"].toList())
        rects.append(rect.toRectF());
    return rects;
}

void File::appendRect(const QRectF &rect)
{
    QList<QVariant> newRects = m_metadata["Rects"].toList();
    newRects.append(rect);
    m_metadata["Rects"] = newRects;
}

void File::appendRect(const cv::Rect &rect)
{
    appendRect(OpenCVUtils::fromRect(rect));
}

void File::appendRects(const QList<QRectF> &rects)
{
    QList<QVariant> newRects = m_metadata["Rects"].toList();
    foreach (const QRectF &rect, rects)
        newRects.append(rect);
    m_metadata["Rects"] = newRects;
}

void File::appendRects(const QList<cv::Rect> &rects)
{
    appendRects(OpenCVUtils::fromRects(rects));
}

QList<RotatedRect> File::namedRotatedRects() const
{
    QList<RotatedRect> rects;
    foreach (const QString &key, localMetadata().keys()) {
        const QVariant &variant = m_metadata[key];
        if (variant.canConvert<RotatedRect>())
            rects.append(variant.value<RotatedRect>());
    }
    return rects;
}

/* File - private methods */
void File::init(const QString &file)
{
    fte = false;
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
                set(words[0], words[1]);
            }
        }
        name = name.left(index);
    }

    if (contains("FTE")) {
        fte = getBool("FTE");
        remove("FTE");
    }
}

/* File - global methods */
QDebug br::operator<<(QDebug dbg, const File &file)
{
    return dbg.nospace() << qPrintable(file.flat());
}

QDataStream &br::operator<<(QDataStream &stream, const File &file)
{
    File temp = file;
    temp.set("FTE",QVariant::fromValue(file.fte));
    return stream << temp.name << temp.m_metadata;
}

QDataStream &br::operator>>(QDataStream &stream, File &file)
{
    stream >> file.name >> file.m_metadata;
    file.fte = file.getBool("FTE", false);
    return stream;
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

QList<int> FileList::crossValidationPartitions() const
{
    QList<int> crossValidationPartitions; crossValidationPartitions.reserve(size());
    foreach (const File &f, *this)
        crossValidationPartitions.append(f.get<int>("Partition", 0));
    return crossValidationPartitions;
}

int FileList::failures() const
{
    int failures = 0;
    foreach (const File &file, *this)
        if (file.fte)
            failures++;
    return failures;
}

/* Template - global methods */
template <typename T>
static T findAndRemove(QVariantMap &map, const QString &key, const T &defaultValue)
{
    T result = defaultValue;
    if (map.contains(key)) {
        result = map.value(key).value<T>();
        map.remove(key);
    }
    return result;
}

br_utemplate Template::toUniversalTemplate(const Template &t)
{
    QVariantMap map = t.file.localMetadata();

    // QJsonObject::fromVariantMap (below) fails to convert
    // QRects and QPoints to string, replacing them with null values.
    // so hand-convert these weirdos
    foreach (const QString &k, map.keys()) {
        QVariant v = map[k];
        if (v.canConvert(QVariant::PointF) || v.canConvert(QVariant::RectF)) {
            QString newv = QtUtils::toString(v);
            map[k] = newv;
        }
        // lists of points and rects, too
        else if (v.type() == QVariant::List) {
            QVariantList oldlist = qvariant_cast<QVariantList>(v);
            if (!oldlist.isEmpty() && (oldlist.first().canConvert(QVariant::PointF) || oldlist.first().canConvert(QVariant::RectF))) {
                QVariantList newlist;
                foreach (const QVariant &subv, oldlist) {
                    newlist.append(QtUtils::toString(subv));
                }
                map[k] = newlist;
            }
        }
    }

    const int32_t  algorithmID = findAndRemove<int32_t> (map, "AlgorithmID", 0);
    const uint32_t frame       = findAndRemove<uint32_t>(map, "Frame"      , std::numeric_limits<uint32_t>::max());
    const int32_t  x           = findAndRemove<int32_t> (map, "X"          , 0);
    const int32_t  y           = findAndRemove<int32_t> (map, "Y"          , 0);
    const uint32_t width       = findAndRemove<uint32_t>(map, "Width"      , 0);
    const uint32_t height      = findAndRemove<uint32_t>(map, "Height"     , 0);
    const float    confidence  = findAndRemove<float>   (map, "Confidence" , 0);
    const uint32_t personID    = findAndRemove<uint32_t>(map, "PersonID"   , std::numeric_limits<uint32_t>::max());
    const QByteArray metadata = QJsonDocument(QJsonObject::fromVariantMap(map)).toJson();
    const Mat &m = t;
    return br_new_utemplate(algorithmID, frame, x, y, width, height, confidence, personID, metadata.constData(), (const char*) m.data, m.rows * m.cols * m.elemSize());
}

Template Template::fromUniversalTemplate(br_const_utemplate ut)
{
    QVariantMap map = QJsonDocument::fromJson(QByteArray((const char*) ut->data)).object().toVariantMap();

    // QJsonDocument::fromJson doesn't know about QRects and QPoints
    // so convert any QStrings that can be converted
    foreach (const QString &k, map.keys()) {
        QVariant v = map[k];
        QVariant newv;
        bool istype;
        if (v.type() == QVariant::String) {
            QString vstr = qvariant_cast<QString>(v);
            newv = QtUtils::toRect(vstr, &istype);
            if (!istype) {
                newv = QtUtils::toPoint(vstr, &istype);
                if (!istype) {
                    newv = v;
                }
            }
            map[k] = newv;
        }
        // convert lists of rects and points, too
        else if (v.type() == QVariant::List) {
            QVariantList oldlist = qvariant_cast<QVariantList>(v);
            if (!oldlist.isEmpty() && oldlist.first().type() == QVariant::String) {
                QString test = qvariant_cast<QString>(oldlist.first());
                QtUtils::toRect(test, &istype);
                QVariantList newlist;
                if (istype) {
                    foreach (const QVariant &subv, oldlist) {
                        newlist.append(QtUtils::toRect(qvariant_cast<QString>(subv)));
                    }
                } else {
                    QtUtils::toPoint(test, &istype);
                    if (istype) {
                        foreach (const QVariant &subv, oldlist) {
                            newlist.append(QtUtils::toPoint(qvariant_cast<QString>(subv)));
                        }
                    } else {
                        newlist = oldlist;
                    }
                }
                map[k] = newlist;
            }
        }
    }

    map.insert("AlgorithmID", ut->algorithmID);
    map.insert("Frame"      , ut->frame      );
    map.insert("X"          , ut->x          );
    map.insert("Y"          , ut->y          );
    map.insert("Width"      , ut->width      );
    map.insert("Height"     , ut->height     );
    map.insert("Confidence" , ut->confidence );
    map.insert("PersonID"   , ut->personID   );
    const Mat m = Mat(1, ut->fvSize, CV_8UC1, (void*)(ut->data + ut->mdSize)).clone();
    return Template(File(map), m);
}

br_utemplate Template::readUniversalTemplate(QFile &file)
{
    const size_t headerSize = sizeof(br_universal_template);
    br_universal_template *t = (br_universal_template*) malloc(headerSize);
    file.read((char*) t, headerSize);

    const size_t dataSize = t->mdSize + t->fvSize;
    t = (br_universal_template*) realloc(t, headerSize + dataSize);
    file.read((char*) &t->data, dataSize);
    return t;
}

void Template::writeUniversalTemplate(QFile &file, br_const_utemplate t)
{
    const qint64 size = sizeof(br_universal_template) + t->mdSize + t->fvSize;
    if (file.write((const char *) t, size) != size)
        qFatal("Failed to write universal template!");
}

void Template::freeUniversalTemplate(br_const_utemplate t)
{
    free((void*) t);
}

QDataStream &br::operator<<(QDataStream &stream, const Template &t)
{
    return stream << static_cast<const QList<cv::Mat>&>(t) << t.file;
}

QDataStream &br::operator>>(QDataStream &stream, Template &t)
{
    return stream >> static_cast<QList<cv::Mat>&>(t) >> t.file;
}

/* TemplateList - public methods */
TemplateList TemplateList::fromGallery(const br::File &gallery, bool partition)
{
    TemplateList templates;
    foreach (const br::File &file, gallery.split()) {
        QScopedPointer<Gallery> i(Gallery::make(file));
        TemplateList newTemplates = i->read();

        // If file is a Format not a Gallery (e.g. XML Format vs. XML Gallery)
        if (newTemplates.isEmpty())
            newTemplates.append(file);

        newTemplates = newTemplates.mid(gallery.get<int>("pos", 0), gallery.get<int>("length", -1));

        const int step = gallery.get<int>("step", 1);
        if (step > 1) {
            TemplateList downsampled; downsampled.reserve(newTemplates.size()/step);
            for (int i=0; i<newTemplates.size(); i+=step)
                downsampled.append(newTemplates[i]);
            newTemplates = downsampled;
        }

        if (gallery.getBool("reduce"))
            newTemplates = newTemplates.reduced();

        if (abs(Globals->crossValidate) > 1 && partition)
            newTemplates = newTemplates.partition("Label");

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

TemplateList TemplateList::fromBuffer(const QByteArray &buffer)
{
    TemplateList templateList;
    QDataStream stream(buffer);
    while (!stream.atEnd()) {
        Template t;
        stream >> t;
        templateList.append(t);
    }
    return templateList;
}

// indexes some property, assigns an integer id to each unique value of propName
// stores the index values in "Label" of the output template list
TemplateList TemplateList::relabel(const TemplateList &tl, const QString &propName, bool preserveIntegers)
{
    const QList<QString> originalLabels = File::get<QString>(tl, propName);
    QHash<QString,int> labelTable;
    foreach (const QString &label, originalLabels)
        if (!labelTable.contains(label)) {
            int value; bool ok;
            value = label.toInt(&ok);
            // If the label is already an integer value we don't want to change it.
            if (ok && preserveIntegers) labelTable.insert(label, value);
            else                        labelTable.insert(label, labelTable.size());
        }

    TemplateList result = tl;
    for (int i=0; i<result.size(); i++)
        result[i].file.set("Label", labelTable[originalLabels[i]]);
    return result;
}

QList<int> TemplateList::indexProperty(const QString &propName, QHash<QString, int> * valueMap,QHash<int, QVariant> * reverseLookup) const
{
    QHash<QString, int> dummyForwards;
    QHash<int, QVariant> dummyBackwards;

    if (!valueMap) valueMap = &dummyForwards;
    if (!reverseLookup) reverseLookup = &dummyBackwards;

    return indexProperty(propName, *valueMap, *reverseLookup);
}

QList<int> TemplateList::indexProperty(const QString &propName, QHash<QString, int> &valueMap, QHash<int, QVariant> &reverseLookup) const
{
    valueMap.clear();
    reverseLookup.clear();

    const QList<QVariant> originalLabels = File::values(*this, propName);
    foreach (const QVariant &label, originalLabels) {
        QString labelString = label.toString();
        if (!valueMap.contains(labelString)) {
            reverseLookup.insert(valueMap.size(), label);
            valueMap.insert(labelString, valueMap.size());
        }
    }

    QList<int> result;
    for (int i=0; i<originalLabels.size(); i++)
        result.append(valueMap[originalLabels[i].toString()]);

    return result;
}

// uses -1 for missing values
QList<int> TemplateList::applyIndex(const QString &propName, const QHash<QString, int> &valueMap) const
{
    const QList<QString> originalLabels = File::get<QString>(*this, propName);

    QList<int> result;
    for (int i=0; i<originalLabels.size(); i++) {
        if (!valueMap.contains(originalLabels[i])) result.append(-1);
        else result.append(valueMap[originalLabels[i]]);
    }

    return result;
}

TemplateList TemplateList::partition(const QString &inputVariable, bool random, bool overwrite) const
{
    const int crossValidate = std::abs(Globals->crossValidate);
    if (crossValidate < 2)
        return *this;

    TemplateList partitioned = *this;

    if (Globals->verbose)
        qDebug() << "Total templates before partition:" << partitioned.size();

    for (int i=partitioned.size()-1; i>=0; i--) {
        // See CrossValidateTransform for description of these variables
        if (partitioned[i].file.getBool("duplicatePartitions")) {
            partitioned[i].file.set("Partition", QVariant(0));
            for (int j=crossValidate-1; j>0; j--) {
                Template duplicateTemplate = partitioned[i].clone();
                duplicateTemplate.file.set("Partition", j);
                partitioned.insert(i+1, duplicateTemplate);
            }
        } else if (partitioned[i].file.getBool("allPartitions")) {
            partitioned[i].file.set("Partition", -1);
        } else {
            if (partitioned[i].file.contains(inputVariable)) {
                const QByteArray md5 = QCryptographicHash::hash(partitioned[i].file.get<QString>(inputVariable).toLatin1(), QCryptographicHash::Md5);
                if (random) {
                    partitioned[i].file.set("Partition", rand() % crossValidate);
                } else if (!partitioned[i].file.contains("Partition") || overwrite) {
                    // Select the right 8 hex characters so that it can be represented as a 64 bit integer without overflow
                    partitioned[i].file.set("Partition", md5.toHex().right(8).toULongLong(0, 16) % crossValidate);
                }
            } else if (Globals->verbose)
                qDebug() << QString("Template does not contain %1 key/value pair used to partition!").arg(inputVariable);
        }
    }

    if (Globals->verbose)
        qDebug() << "Total templates after partition:" << partitioned.size();

    return partitioned;
}

/* Object - public methods */
QStringList Object::parameters() const
{
    QStringList parameters;

    for (int i = firstAvailablePropertyIdx; i < metaObject()->propertyCount();i++) {
        QMetaProperty property = metaObject()->property(i);
        parameters.append(QString("%1 %2 = %3").arg(property.typeName(), property.name(), property.read(this).toString()));
    }

    return parameters;
}

QStringList Object::prunedArguments(bool expanded) const
{
    QStringList arguments;
    QString className = this->metaObject()->className();
    QScopedPointer<Object> shellObject;

    if (className.startsWith("br::"))
        className = className.mid(4);
    if (!className.startsWith("."))
        className = "." + className;

    const QMetaObject *interface = getInterface(this);
    QString interfaceName = QString(interface->className()).remove("br::");

    if (className.endsWith(interfaceName))
        className.chop(interfaceName.size());

    if (interfaceName == "Representation")
        shellObject.reset(Factory<Representation>::make(className));
    else if (interfaceName == "Classifier")
        shellObject.reset(Factory<Classifier>::make(className));
    else if (interfaceName == "Distance")
        shellObject.reset(Factory<Distance>::make(className));
    else if (interfaceName == "Transform")
        shellObject.reset(Factory<Transform>::make(className));
    else if (interfaceName == "Format")
        shellObject.reset(Factory<Format>::make(className));
    else if (interfaceName == "Initializer")
        shellObject.reset(Factory<Initializer>::make(className));
    else if (interfaceName == "Output")
        shellObject.reset(Factory<Output>::make(className));
    else
        qFatal("Object with className: %s has unrecognized interface: %s", qPrintable(className), qPrintable(interfaceName));

    for (int i=firstAvailablePropertyIdx; i<metaObject()->propertyCount(); i++) {
        const char *name = metaObject()->property(i).name();

        QVariant defaultVal = shellObject->property(name);

        if (defaultVal != property(name))
            arguments.append(name + QString("=") + argument(i, expanded));
    }

    return arguments;
}

QString Object::argument(int index, bool expanded) const
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
                strings.append(transform->description(expanded));
        } else if (type == "QList<br::Distance*>") {
            foreach (Distance *distance, variant.value< QList<Distance*> >())
                strings.append(distance->description(expanded));
        } else if (type == "QList<br::Representation*>") {
            foreach (Representation *representation, variant.value< QList<Representation*> >())
                strings.append(representation->description(expanded));
        } else if (type == "QList<br::Classifier*>") {
            foreach (Classifier *classifier, variant.value< QList<Classifier*> >())
                strings.append(classifier->description(expanded));
        } else {
            qFatal("Unrecognized type: %s", qPrintable(type));
        }

        return "[" + strings.join(",") + "]";
    } else if (type == "br::Transform*") {
        return variant.value<Transform*>()->description(expanded);
    } else if (type == "br::Distance*") {
        return variant.value<Distance*>()->description(expanded);
    } else if (type == "br::Representation*") {
        return variant.value<Representation*>()->description(expanded);
    } else if (type == "br::Classifier*") {
        return variant.value<Classifier*>()->description(expanded);
    } else if (type == "QStringList") {
        return "[" + variant.toStringList().join(",") + "]";
    }

    return variant.toString();
}

QString Object::description(bool expanded) const
{
    QString argumentString = prunedArguments(expanded).join(",");
    if (argumentString.endsWith(","))
        argumentString.chop(1);

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
        } else if (type == "QList<br::Representation*>") {
            foreach (Representation *representation, property.read(this).value< QList<Representation*> >())
                representation->store(stream);
        } else if (type == "QList<br::Classifier*>") {
            foreach (Classifier *classifier, property.read(this).value< QList<Classifier*> >())
                classifier->store(stream);
        } else if (type == "br::Transform*") {
            property.read(this).value<Transform*>()->store(stream);
        } else if (type == "br::Distance*") {
            property.read(this).value<Distance*>()->store(stream);
        } else if (type == "br::Representation*") {
            property.read(this).value<Representation*>()->store(stream);
        } else if (type == "br::Classifier*") {
            property.read(this).value<Classifier*>()->store(stream);
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
        } else if (type == "QList<br::Representation*>") {
            foreach (Representation *representation, property.read(this).value< QList<Representation*> >())
                representation->load(stream);
        } else if (type == "QList<br::Classifier*>") {
            foreach (Classifier *classifier, property.read(this).value< QList<Classifier*> >())
                classifier->load(stream);
        } else if (type == "br::Transform*") {
            property.read(this).value<Transform*>()->load(stream);
        } else if (type == "br::Distance*") {
            property.read(this).value<Distance*>()->load(stream);
        } else if (type == "br::Representation*") {
            property.read(this).value<Representation*>()->load(stream);
        } else if (type == "br::Classifier*") {
            property.read(this).value<Classifier*>()->load(stream);
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

bool Object::setExistingProperty(const QString &name, QVariant value)
{
    if (this->metaObject()->indexOfProperty(qPrintable(name)) == -1)
        return false;
    setProperty(name, value);
    init();
    return true;
}

QList<Object *> Object::getChildren() const
{
    QList<Object *> output;
    for (int i=0; i < metaObject()->propertyCount(); i++) {
        const char *prop_name = metaObject()->property(i).name();
        const QVariant &variant = this->property(prop_name);

        if (variant.canConvert<Transform *>()) {
            Transform *tform = variant.value<Transform *>();
            if (tform)
                output.append((Object* ) variant.value<Transform *>());
        }
        else if (variant.canConvert<QList<Transform *> >()) {
            foreach (const Transform *tform, variant.value<QList<Transform *> >()) {
                if (tform)
                    output.append((Object *) tform);
            }
        }
        else if (variant.canConvert<Distance *>()) {
            Distance *dist = variant.value<Distance *>();
            if (dist)
                output.append((Object* ) variant.value<Distance *>());
        }
        else if (variant.canConvert<QList<Distance *> >()) {
            foreach (const Distance *dist, variant.value<QList<Distance *> >()) {
                if (dist)
                    output.append((Object *) dist);
            }
        }
        else if (variant.canConvert<Classifier *>()) {
            Classifier *classifier = variant.value<Classifier *>();
            if (classifier)
                output.append((Object* ) variant.value<Classifier *>());
        }
        else if (variant.canConvert<QList<Classifier *> >()) {
            foreach (const Classifier *classifier, variant.value<QList<Classifier *> >()) {
                if (classifier)
                    output.append((Object *) classifier);
            }
        }
    }
    return output;
}

bool Object::setPropertyRecursive(const QString &name, QVariant value)
{
    // collect children
    bool res = setExistingProperty(name, value);
    if (res)
        return true;

    QList<Object *> children = getChildren();
    foreach (Object *obj, children) {
        if (obj->setPropertyRecursive(name, value)) {
            init();
            return true;
        }
    }
    return false;
}

void Object::setProperty(const QString &name, QVariant value)
{
    QString type;
    int index = metaObject()->indexOfProperty(qPrintable(name));
    if (index != -1) type = metaObject()->property(index).typeName();

    if (metaObject()->property(index).isEnumType()) {
        // This is necessary because setProperty can only set enums
        // using their integer value if the QVariant is of type int (or uint)
        bool ok;
        int v = value.toInt(&ok);
        if (ok)
            value = v;
    } else if ((type.startsWith("QList<") && type.endsWith(">")) || (type == "QStringList") || (type == "QVariantList")) {
        QVariantList elements;
        if (value.canConvert< QList<Transform*> >()) {
            foreach (Transform *transform, value.value< QList<Transform*> >())
                elements.append(QVariant::fromValue(transform));
        } else if (value.canConvert<QString>()) {
            QString string = value.value<QString>();
            if (!string.startsWith('[') || !string.endsWith(']'))
                qFatal("Expected a list to start with '[' and end with 'brackets']'.");
            foreach (const QString &element, parse(string.mid(1, string.size()-2)))
                elements.append(element);
        } else if (value.canConvert<QVariantList>()) {
            elements = value.value<QVariantList>();
        } else {
            qFatal("Expected a list.");
        }

        if ((type == "QList<QVariant>") || (type == "QVariantList")) {
            value.setValue(elements);
        } else if ((type == "QList<QString>") || (type == "QStringList")) {
            QStringList parsedValues;
            foreach (const QVariant &element, elements) {
                if (element.canConvert<QString>()) {
                    parsedValues.append(element.toString());
                } else if (element.canConvert<QPointF>()) {
                    const QPointF point = element.toPointF();
                    parsedValues.push_back(QString("(%1,%2)").arg(QString::number(point.x()), QString::number(point.y())));
                } else if (element.canConvert<QRectF>()) {
                    const QRectF rect = element.toRectF();
                    parsedValues.push_back(QString("(%1,%2,%3,%4)").arg(QString::number(rect.x()), QString::number(rect.y()), QString::number(rect.width()), QString::number(rect.height())));
                } else {
                    qFatal("Can't convert variant to string.");
                }
            }
            value.setValue(parsedValues);
        } else if (type == "QList<float>") {
            QList<float> parsedValues; bool ok;
            foreach (const QVariant &element, elements) {
                parsedValues.append(element.toFloat(&ok));
                if (!ok) qFatal("Failed to convert element to floating point format.");
            }
            value.setValue(parsedValues);
        } else if (type == "QList<int>") {
            QList<int> parsedValues; bool ok;
            foreach (const QVariant &element, elements) {
                parsedValues.append(element.toInt(&ok));
                if (!ok) qFatal("Failed to convert element to integer format.");
            }
            value.setValue(parsedValues);
        } else if (type == "QList<br::Transform*>") {
            QList<Transform*> parsedValues;
            foreach (const QVariant &element, elements)
                if (element.canConvert<QString>()) parsedValues.append(Transform::make(element.toString(), this));
                else                               parsedValues.append(element.value<Transform*>());
            value.setValue(parsedValues);
        } else if (type == "QList<br::Distance*>") {
            QList<Distance*> parsedValues;
            foreach (const QVariant &element, elements)
                if (element.canConvert<QString>()) parsedValues.append(Distance::make(element.toString(), this));
                else                               parsedValues.append(element.value<Distance*>());
            value.setValue(parsedValues);
        } else if (type == "QList<br::Representation*>") {
            QList<Representation*> parsedValues;
            foreach (const QVariant &element, elements)
                if (element.canConvert<QString>()) parsedValues.append(Representation::make(element.toString(), this));
                else                               parsedValues.append(element.value<Representation*>());
            value.setValue(parsedValues);
        } else if (type == "QList<br::Classifier*>") {
            QList<Classifier*> parsedValues;
            foreach (const QVariant &element, elements)
                if (element.canConvert<QString>()) parsedValues.append(Classifier::make(element.toString(), this));
                else                               parsedValues.append(element.value<Classifier*>());
            value.setValue(parsedValues);
        } else {
            qFatal("Unrecognized type: %s", qPrintable(type));
        }
    } else if (type == "br::Transform*") {
        if (value.canConvert<QString>())
            value.setValue(Transform::make(value.toString(), this));
    } else if (type == "br::Distance*") {
        if (value.canConvert<QString>())
            value.setValue(Distance::make(value.toString(), this));
    } else if (type == "br::Representation*") {
        if (value.canConvert<QString>())
            value.setValue(Representation::make(value.toString(), this));
    } else if (type == "br::Classifier*") {
        if (value.canConvert<QString>())
            value.setValue(Classifier::make(value.toString(), this));
    } else if (type == "bool") {
        if      (value.isNull())   value = true;
        else if (value == "false") value = false;
        else if (value == "true")  value = true;
    } else if (type == "cv::Mat") {
        if (value.toString().isEmpty())
            value.setValue(cv::Mat());
        else
            qFatal("QString to cv::Mat not implemented!");
    }

    if (!QObject::setProperty(qPrintable(name), value) && !type.isEmpty())
        qFatal("Failed to set %s %s::%s to: %s",
               qPrintable(type), metaObject()->className(), qPrintable(name), qPrintable(value.toString()));
}

QStringList Object::parse(const QString &string, char split)
{
    return QtUtils::parse(string, split);
}

/* Object - private methods */
void Object::init(const File &file_)
{
    file = file_;

    const QMetaObject *interface = getInterface(this);
    if (strcmp(interface->className(), metaObject()->className()) != 0)
        firstAvailablePropertyIdx = interface->propertyCount();
    else
        firstAvailablePropertyIdx = interface->propertyOffset();

    QString interfaceName = QString(interface->className()).remove("br::");

    // Strip interface name from object name (e.g. PipeTransform -> Pipe)
    QString name = QString(metaObject()->className()).remove("br::");
    if (name.endsWith(interfaceName)) name = name.left(name.size() - interfaceName.size());
    setObjectName(name);

    // Set properties to their default values
    for (int i=0; i<metaObject()->propertyCount(); i++) {
        QMetaProperty property = metaObject()->property(i);
        if (property.isResettable())
            if (!property.reset(this))
                qFatal("Failed to reset %s::%s", metaObject()->className(), property.name());
    }

    foreach (QString key, file.localKeys()) {
        const QVariant value = file.value(key);
        if (key.startsWith(("_Arg"))) {
            int argumentNumber =  key.mid(4).toInt();
            int targetIdx = argumentNumber + firstAvailablePropertyIdx;
            if (targetIdx >= metaObject()->propertyCount()) {
                qWarning("Too many arguments for object: %s, ignoring: %s", qPrintable(objectName()), qPrintable(value.toString()));
                continue;
            }
            key = metaObject()->property(targetIdx).name();
        }
        setProperty(key, value);
    }

    init();
}

/* Context - public methods */
bool br::Context::contains(const QString &name)
{
    return property(qPrintable(name)).isValid();
}

void br::Context::printStatus()
{
    if (verbose || quiet || (totalSteps < 2)) return;
    const float p = progress();
    if (p < 1) {
        int s = timeRemaining();
        fprintf(stderr,"\r%05.2f%%  ELAPSED=%s  REMAINING=%s  COUNT=%g", p*100, QtUtils::toTime(Globals->startTime.elapsed()/1000.0f).toStdString().c_str(), QtUtils::toTime(s).toStdString().c_str(), Globals->currentStep);
        fflush(stderr);
    }
}

float br::Context::progress() const
{
    if (totalSteps == 0) return -1;
    return currentProgress / totalSteps;
}

void br::Context::setProperty(const QString &key, const QString &value)
{
    Object::setProperty(key, value.isEmpty() ? QVariant(true) : value);
    qDebug("Set %s%s", qPrintable(key), value.isEmpty() ? "" : qPrintable(" to " + value));

    if (key == "parallelism") {
        if (parallelism != 0)
            QThreadPool::globalInstance()->setMaxThreadCount(abs(parallelism));
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

// We create our own when the user hasn't.
// Since we can't ensure that it gets deleted last, we never delete it.
static QCoreApplication *application = NULL;

void br::Context::initialize(int &argc, char **argv, QString sdkPath, bool useGui)
{
    QString sep;
#ifndef _WIN32
  #ifndef __APPLE__
    // Modern OS X will only define the DISPLAY environment variable if XQuartz
    // is installed, so we only do this check on non-Apple Unix systems.
    useGui = useGui && (getenv("DISPLAY") != NULL);
  #endif // __APPLE__
    sep = ":";
#else // _WIN32
    sep = ";";
#endif // not _WIN32

    // We take in argc as a reference due to:
    //   https://bugreports.qt-project.org/browse/QTBUG-5637
    // QApplication should be initialized before anything else.
    if (QCoreApplication::instance() == NULL) {
#ifndef BR_EMBEDDED
        if (useGui) application = new QApplication(argc, argv);
        else        application = new QCoreApplication(argc, argv);
#else // not BR_EMBEDDED
        useGui = false;
        application = new QCoreApplication(argc, argv);
#endif // BR_EMBEDDED
    }

    // Search for SDK
    if (sdkPath.isEmpty()) {
        QStringList checkPaths; checkPaths << QCoreApplication::applicationDirPath() << QDir::currentPath();
        checkPaths << QString(getenv("PATH")).split(sep, QString::SkipEmptyParts);
        QSet<QString> checkedPaths; // Avoid infinite loops from symlinks

        bool foundSDK = false;
        foreach (const QString &path, checkPaths) {
            if (foundSDK) break;
            QDir dir(path);
            do {
                sdkPath = dir.absolutePath();
                if (checkedPaths.contains(sdkPath)) break;
                else                                checkedPaths.insert(sdkPath);
                foundSDK = checkSDKPath(sdkPath);
                dir.cdUp();
            } while (!foundSDK && !dir.isRoot());
        }

        if (!foundSDK) {
            qWarning("Unable to locate SDK (share/openbr/openbr.bib) automatically from paths: %s", qPrintable(checkPaths.join("\n")));
            return;
        }
    } else {
        if (!checkSDKPath(sdkPath)) {
            qWarning("Unable to locate SDK from %s.", qPrintable(sdkPath));
            return;
        }
    }

    QCoreApplication::setOrganizationName(COMPANY_NAME);
    QCoreApplication::setApplicationName(PRODUCT_NAME);
    QCoreApplication::setApplicationVersion(PRODUCT_VERSION);

    qRegisterMetaType<cv::Mat>();
    qRegisterMetaType<cv::RotatedRect>();
    qRegisterMetaType<br::File>();
    qRegisterMetaType<br::FileList>();
    qRegisterMetaType<br::Template>();
    qRegisterMetaType<br::TemplateList>();
    qRegisterMetaType< br::Transform* >();
    qRegisterMetaType< br::Distance* >();
    qRegisterMetaType< br::Representation* >();
    qRegisterMetaType< br::Classifier* >();
    qRegisterMetaType< QList<int> >();
    qRegisterMetaType< QList<float> >();
    qRegisterMetaType< QList<br::Transform*> >();
    qRegisterMetaType< QList<br::Distance*> >();
    qRegisterMetaType< QList<br::Representation* > >();
    qRegisterMetaType< QList<br::Classifier* > >();

    Globals = new Context();
    Globals->init(File());
    Globals->useGui = useGui;
    Globals->algorithm = "Identity";
    Globals->path = getenv("DATA"); // our convention
    Globals->sdkPath = sdkPath;

    // The message handler requires a valid `Globals` so we set it after `Globals` is constructed
    qInstallMessageHandler(messageHandler);

    // We seed with 0 instead of time(NULL) to have reproducible randomness
    srand(0);

    // Disable OpenCV parallelism, we prefer to parallelize at the image level
    setNumThreads(0);

    // Trigger registered initializers
    QList< QSharedPointer<Initializer> > initializers = Factory<Initializer>::makeAll();
    foreach (const QSharedPointer<Initializer> &initializer, initializers)
        initializer->initialize();
}

void br::Context::finalize()
{
    qInstallMessageHandler(0);

    // Trigger registered finalizers
    QList< QSharedPointer<Initializer> > initializers = Factory<Initializer>::makeAll();
    foreach (const QSharedPointer<Initializer> &initializer, initializers)
        initializer->finalize();

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


QStringList br::Context::objects(const char *abstractions, const char *implementations, bool parameters)
{
    QStringList objectList;
    QRegExp abstractionsRegExp(abstractions);
    QRegExp implementationsRegExp(implementations);

    if (abstractionsRegExp.exactMatch("Abbreviation"))
        foreach (const QString &name, Globals->abbreviations.keys())
            if (implementationsRegExp.exactMatch(name))
                objectList.append(name + (parameters ? "\t" + Globals->abbreviations[name] : ""));

    if (abstractionsRegExp.exactMatch("Distance"))
        foreach (const QString &name, Factory<Distance>::names())
            if (implementationsRegExp.exactMatch(name))
                objectList.append(name + (parameters ? "\t" + Factory<Distance>::parameters(name) : ""));

    if (abstractionsRegExp.exactMatch("Format"))
        foreach (const QString &name, Factory<Format>::names())
            if (implementationsRegExp.exactMatch(name))
                objectList.append(name + (parameters ? "\t" + Factory<Format>::parameters(name) : ""));

    if (abstractionsRegExp.exactMatch("Initializer"))
        foreach (const QString &name, Factory<Initializer>::names())
            if (implementationsRegExp.exactMatch(name))
                objectList.append(name + (parameters ? "\t" + Factory<Initializer>::parameters(name) : ""));

    if (abstractionsRegExp.exactMatch("Output"))
        foreach (const QString &name, Factory<Output>::names())
            if (implementationsRegExp.exactMatch(name))
                objectList.append(name + (parameters ? "\t" + Factory<Output>::parameters(name) : ""));

    if (abstractionsRegExp.exactMatch("Transform"))
        foreach (const QString &name, Factory<Transform>::names())
            if (implementationsRegExp.exactMatch(name))
                objectList.append(name + (parameters ? "\t" + Factory<Transform>::parameters(name) : ""));

    if (abstractionsRegExp.exactMatch("Representation"))
        foreach (const QString &name, Factory<Representation>::names())
            if (implementationsRegExp.exactMatch(name))
                objectList.append(name + (parameters ? "\t" + Factory<Representation>::parameters(name) : ""));

    if (abstractionsRegExp.exactMatch("Classifier"))
        foreach (const QString &name, Factory<Classifier>::names())
            if (implementationsRegExp.exactMatch(name))
                objectList.append(name + (parameters ? "\t" + Factory<Classifier>::parameters(name) : ""));

    return objectList;
}

void br::Context::messageHandler(QtMsgType type, const QMessageLogContext &context, const QString &msg)
{
    // Something about this method is not thread safe, and will lead to crashes if qDebug
    // statements are called from multiple threads. Unless we lock the whole thing...
    static QMutex generalLock;
    QMutexLocker locker(&generalLock);

    QString txt;
    if (type == QtDebugMsg) {
        if (Globals->quiet) return;
        txt = QString("%1\n").arg(msg);
    } else {
        // Ignore QSslSocket warnings, too much noise and not enough signal
        if ((type == QtWarningMsg) && msg.contains("QSslSocket"))
            return;

        switch (type) {
          case QtWarningMsg:  txt = QString("Warning: %1\n" ).arg(msg); break;
          case QtCriticalMsg: txt = QString("Critical: %1\n").arg(msg); break;
          default:            txt = QString("Fatal: %1\n"   ).arg(msg); break;
        }

        if (Globals->sdkPath != ":")
            txt += "  SDK Path: "  + Globals->sdkPath + "\n";
        if (!QString(context.file).isEmpty())
            txt += "  File: " + QString(context.file) + "\n";
        if (!QString(context.function).isEmpty())
            txt += "  Function: " + QString(context.function) + "\n";
        if (context.line != 0)
            txt += "  Line: " + QString::number(context.line) + "\n";
    }

    std::cerr << txt.toStdString();
    Globals->mostRecentMessage = txt;

    if (Globals->logFile.isWritable()) {
        Globals->logFile.write(qPrintable(txt));
        Globals->logFile.flush();
    }

    if (type == QtFatalMsg) {
#ifdef _WIN32
        QCoreApplication::quit(); // abort() hangs the console on Windows for some reason related to the event loop not being exited
#else // not _WIN32
        abort(); // We abort so we can get a stack trace back to the code that triggered the message.
#endif // _WIN32
    }
}

Context *br::Globals = NULL;

/* Output - public methods */
void Output::initialize(const FileList &targetFiles, const FileList &queryFiles)
{
    this->targetFiles = targetFiles;
    this->queryFiles = queryFiles;
    if (this->blockRows == -1)
        blockRows = Globals->blockSize;

    if (this->blockCols == -1)
        blockCols = Globals->blockSize;

    selfSimilar = (queryFiles == targetFiles) && (targetFiles.size() > 1) && (queryFiles.size() > 1);
}

void Output::setBlock(int rowBlock, int columnBlock)
{
    offset = QPoint((columnBlock == -1) ? 0 : blockCols*columnBlock,
                    (rowBlock == -1) ? 0 : blockRows*rowBlock);
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
    return QString::number(data.at<float>(row,column));
}

/* MatrixOutput - private methods */
void MatrixOutput::set(float value, int i, int j)
{
    data.at<float>(i,j) = value;
}

BR_REGISTER(Output, MatrixOutput)

/* Format - public methods */
Template Format::read(const QString &file)
{
    return QScopedPointer<Format>(Factory<Format>::make(file))->read();
}

void Format::write(const QString &file, const Template &t)
{
    QScopedPointer<Format>(Factory<Format>::make(file))->write(t);
}

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

// Default init -- if the file contains "append", read the existing
// data and immediately write it
void Gallery::init()
{
    if (file.exists() && file.contains("append"))
    {
        File rFile = file;
        rFile.remove("append");
        Gallery *reader = Gallery::make(rFile);
        TemplateList data = reader->read();
        delete reader;

        writeBlock(data);
    }
}

/* Transform - public methods */
Transform::Transform(bool _independent, bool _trainable)
{
    independent = _independent;
    trainable = _trainable;
}

Transform *Transform::make(QString str, QObject *parent)
{
    // Check for custom transforms
    if (Globals->abbreviations.contains(str))
        return make(Globals->abbreviations[str], parent);

    File parsed("."+str);
    if (Globals->abbreviations.contains(parsed.suffix())) {
        Transform *res = make(Globals->abbreviations[parsed.suffix()], parent);
        applyAdditionalProperties(parsed, res);
        return res;
    }

    //! [Make a pipe]
    { // Check for use of '+' as shorthand for Pipe(...)
        QStringList words = parse(str, '+');
        if (words.size() > 1)
            return make("Pipe([" + words.join(",") + "])", parent);
    }
    //! [Make a pipe]

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

    // Base name not found? Try constructing it via LoadStore
    if (!Factory<Transform>::names().contains(parsed.suffix())
        && (QFileInfo(parsed.suffix()).exists()
            || QFileInfo(Globals->sdkPath + "/share/openbr/models/transforms/"+parsed.suffix()).exists()
            || QFileInfo(Globals->sdkPath + "/../share/openbr/models/transforms/"+parsed.suffix()).exists())) {
        Transform *tform = make("<"+parsed.suffix()+">", parent);
        applyAdditionalProperties(parsed, tform);
        return tform;
    }

    //! [Construct the root transform]
    Transform *transform = Factory<Transform>::make("." + str);
    //! [Construct the root transform]

    if (transform->independent) {
        File independent(".Independent");
        independent.set("transform", QVariant::fromValue(transform));
        transform = Factory<Transform>::make(independent);
    }

    transform->setParent(parent);
    return transform;
}

Transform *Transform::clone() const
{
   Transform *clone = Factory<Transform>::make("."+description(false));
   return clone;
}

static void _project(const Transform *transform, const Template *src, Template *dst)
{
    try {
        transform->project(*src, *dst);
    } catch (...) {
        qWarning("Exception triggered when processing %s with transform %s", qPrintable(src->file.flat()), qPrintable(transform->objectName()));
        *dst = Template(src->file);
        dst->file.fte = true;
    }
}

// Default project(TemplateList) calls project(Template) separately for each element
void Transform::project(const TemplateList &src, TemplateList &dst) const
{
    dst.reserve(src.size());

    for (int i=0; i<src.size(); i++)
        dst.append(Template());
    QFutureSynchronizer<void> futures;
    for (int i=0; i<dst.size(); i++)
        if ((Globals->parallelism > 1) && (dst.size() > 1)) futures.addFuture(QtConcurrent::run(_project, this, &src[i], &dst[i]));
        else                                                _project(this, &src[i], &dst[i]);
    futures.waitForFinished();
}

TemplateEvent *Transform::getEvent(const QString &name)
{
    foreach (Transform *child, getChildren<Transform>()) {
        TemplateEvent *probe = child->getEvent(name);
        if (probe)
            return probe;
    }

    return NULL;
}

void Transform::train(const TemplateList &data)
{
    if (!trainable) {
        qWarning("Train called on untrainable transform %s", this->metaObject()->className());
        return;
    }
    QList<TemplateList> input;
    input.append(data);
    train(input);
}

void Transform::train(const QList<TemplateList> &data)
{
    TemplateList combined;
    foreach (const TemplateList &set, data)
        combined.append(set);

    train(combined);
}

/* Distance - public methods */
Distance *Distance::make(QString str, QObject *parent)
{
    // Check for custom transforms
    if (Globals->abbreviations.contains(str))
        return make(Globals->abbreviations[str], parent);

    // Check for use of '<...>' as shorthand for LoadStore(...)
    if (str.startsWith('<') && str.endsWith('>'))
        return make("LoadStore(" + str.mid(1, str.size()-2) + ")", parent);

    { // Check for use of '+' as shorthand for Pipe(...)
        QStringList words = parse(str, '+');
        if (words.size() > 1)
            return make("Pipe([" + words.join(",") + "])", parent);
    }

    Distance *const distance = Factory<Distance>::make("." + str);
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
    futures.waitForFinished();
}

QList<float> Distance::compare(const TemplateList &targets, const Template &query) const
{
    QList<float> scores; scores.reserve(targets.size());
    foreach (const Template &target, targets)
        scores.append(compare(target, query));
    return scores;
}

float Distance::compare(const Template &a, const Template &b) const
{
    float similarity = 0;
    int comparisons = 0;
    foreach (const cv::Mat &ma, a) {
        foreach (const cv::Mat &mb, b) {
            const float score = compare(ma, mb);
            if (score != -std::numeric_limits<float>::max()) {
                similarity += score;
                comparisons++;
            }
        }
    }

    if (comparisons > 0) similarity /= comparisons;
    else                 similarity = -std::numeric_limits<float>::max();
    return similarity;
}

float Distance::compare(const cv::Mat &a, const cv::Mat &b) const
{
    if (a.empty() || b.empty() || a.rows != b.rows || a.cols != b.cols || a.elemSize() != b.elemSize())
        return -std::numeric_limits<float>::max();

    return compare(a.data, b.data, a.rows * a.cols * a.elemSize());
}

float Distance::compare(const uchar *, const uchar *, size_t) const
{
    qFatal("Logic error: %s did not implement a comparison function or was accessed at an unsupported level of abstraction.", metaObject()->className());
    return -std::numeric_limits<float>::max();
}

/* Distance - private methods */
void Distance::compareBlock(const TemplateList &target, const TemplateList &query, Output *output, int targetOffset, int queryOffset) const
{
    for (int i=0; i<query.size(); i++)
        for (int j=0; j<target.size(); j++)
            if (target[j].isEmpty() || query[i].isEmpty()) output->setRelative(-std::numeric_limits<float>::max(),i+queryOffset, j+targetOffset);
            else output->setRelative(compare(target[j], query[i]), i+queryOffset, j+targetOffset);
}

void br::applyAdditionalProperties(const File &temp, Transform *target)
{
    QVariantMap meta = temp.localMetadata();
    for (QVariantMap::iterator i = meta.begin(); i != meta.end(); ++i) {
        if (i.key().startsWith("_Arg"))
            continue;

        target->setPropertyRecursive(i.key(), i.value() );
    }
}

Transform *br::wrapTransform(Transform *base, const QString &target)
{
    Transform *res = Transform::make(target, NULL);
    res->setPropertyRecursive("transform", QVariant::fromValue(base));
    return res;
}

Transform *br::pipeTransforms(QList<Transform *> &transforms)
{
    Transform *res = Transform::make("Pipe",NULL);
    res->setPropertyRecursive("transforms", QVariant::fromValue(transforms));
    return res;
}

Representation *Representation::make(QString str, QObject *parent)
{
    // Check for custom transforms
    if (Globals->abbreviations.contains(str))
        return make(Globals->abbreviations[str], parent);

    File f = "." + str;
    Representation *rep = Factory<Representation>::make(f);

    rep->setParent(parent);
    return rep;
}

Classifier *Classifier::make(QString str, QObject *parent)
{
    // Check for custom transforms
    if (Globals->abbreviations.contains(str))
        return make(Globals->abbreviations[str], parent);

    File f = "." + str;
    Classifier *classifier = Factory<Classifier>::make(f);

    classifier->setParent(parent);
    return classifier;
}
