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

#ifndef BR_OPENBR_PLUGIN_H
#define BR_OPENBR_PLUGIN_H

#ifdef __cplusplus

#include <QDataStream>
#include <QDebug>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFuture>
#include <QHash>
#include <QList>
#include <QMap>
#include <QPoint>
#include <QPointF>
#include <QRectF>
#include <QScopedPointer>
#include <QSharedPointer>
#include <QString>
#include <QStringList>
#include <QThread>
#include <QTime>
#include <QVariant>
#include <QVector>
#include <opencv2/core/core.hpp>
#include <openbr/openbr.h>
#include <openbr/universal_template.h>
#include <assert.h>

namespace br
{

#define BR_PROPERTY(TYPE,NAME,DEFAULT)                  \
TYPE NAME;                                              \
TYPE get_##NAME() const { return NAME; }                \
void set_##NAME(TYPE the_##NAME) { NAME = the_##NAME; } \
void reset_##NAME() { NAME = DEFAULT; }

struct BR_EXPORT File
{
    QString name;

    File() { fte = false; }
    File(const QString &file) { init(file); }
    File(const QString &file, const QVariant &label) { init(file); set("Label", label); }
    File(const char *file) { init(file); }
    File(const QVariantMap &metadata) : fte(false), m_metadata(metadata) {}
    inline operator QString() const { return name; }
    QString flat() const;
    QString hash() const;

    inline QStringList localKeys() const { return m_metadata.keys(); }
    inline QVariantMap localMetadata() const { return m_metadata; }

    void append(const QVariantMap &localMetadata);
    void append(const File &other);
    inline File &operator+=(const QMap<QString,QVariant> &other) { append(other); return *this; }
    inline File &operator+=(const File &other) { append(other); return *this; }

    QList<File> split() const;
    QList<File> split(const QString &separator) const;

    inline void setParameter(int index, const QVariant &value) { set("_Arg" + QString::number(index), value); }
    inline bool containsParameter(int index) const { return contains("_Arg" + QString::number(index)); }
    inline QVariant getParameter(int index) const { return get<QVariant>("_Arg" + QString::number(index)); }

    inline bool operator==(const char* other) const { return name == other; }
    inline bool operator==(const File &other) const { return (name == other.name) && (m_metadata == other.m_metadata); }
    inline bool operator!=(const File &other) const { return !(*this == other); }
    inline bool operator<(const File &other) const { return name < other.name; }
    inline bool operator<=(const File &other) const { return name <= other.name; }
    inline bool operator>(const File &other) const { return name > other.name; }
    inline bool operator>=(const File &other) const { return name >= other.name; }

    inline bool isNull() const { return name.isEmpty() && m_metadata.isEmpty(); }
    inline bool isTerminal() const { return name == "terminal"; }
    inline bool exists() const { return QFileInfo(name).exists(); } 
    inline QString fileName() const { return QFileInfo(name).fileName(); }
    inline QString baseName() const { const QString baseName = QFileInfo(name).baseName();
                                      return baseName.isEmpty() ? QDir(name).dirName() : baseName; }
    inline QString suffix() const { return QFileInfo(name).suffix(); }
    inline QString path() const { return QFileInfo(name).path(); }
    QString resolved() const;

    bool contains(const QString &key) const;
    bool contains(const QStringList &keys) const;
    QVariant value(const QString &key) const;
    static QVariant parse(const QString &value);
    inline void set(const QString &key, const QVariant &value) { m_metadata.insert(key, value); }
    void set(const QString &key, const QString &value);


    template <typename T>
    void setList(const QString &key, const QList<T> &value)
    {
        QVariantList variantList;
        variantList.reserve(value.size());
        foreach (const T &item, value)
            variantList << item;
        set(key, variantList);
    }

    inline void remove(const QString &key) { m_metadata.remove(key); }


    template <typename T>
    T get(const QString &key) const
    {
        if (!contains(key)) qFatal("Missing key: %s in: %s", qPrintable(key), qPrintable(flat()));
        QVariant variant = value(key);
        if (!variant.canConvert<T>()) qFatal("Can't convert: %s in: %s", qPrintable(key), qPrintable(flat()));
        return variant.value<T>();
    }


    template <typename T>
    T get(const QString &key, const T &defaultValue) const
    {
        if (!contains(key)) return defaultValue;
        QVariant variant = value(key);
        if (!variant.canConvert<T>()) return defaultValue;
        return variant.value<T>();
    }


    bool getBool(const QString &key, bool defaultValue = false) const;


    template <typename T>
    QList<T> getList(const QString &key) const
    {
        if (!contains(key)) qFatal("Missing key: %s in: %s", qPrintable(key), qPrintable(flat()));
        QList<T> list;
        foreach (const QVariant &item, m_metadata[key].toList()) {
            if (item.canConvert<T>()) list.append(item.value<T>());
            else qFatal("Failed to convert value for key %s in: %s", qPrintable(key), qPrintable(flat()));
        }
        return list;
    }


    template <typename T>
    QList<T> getList(const QString &key, const QList<T> defaultValue) const
    {
        if (!contains(key)) return defaultValue;
        QList<T> list;
        foreach (const QVariant &item, m_metadata[key].toList()) {
            if (item.canConvert<T>()) list.append(item.value<T>());
            else return defaultValue;
        }
        return list;
    }


    template<class U>
    static QList<QVariant> values(const QList<U> &fileList, const QString &key)
    {
        QList<QVariant> values; values.reserve(fileList.size());
        foreach (const U &f, fileList) values.append(((const File&)f).value(key));
        return values;
    }


    template<class T, class U>
    static QList<T> get(const QList<U> &fileList, const QString &key)
    {
        QList<T> result; result.reserve(fileList.size());
        foreach (const U &f, fileList) result.append(((const File&)f).get<T>(key));
        return result;
    }


    template<class T, class U>
    static QList<T> get(const QList<U> &fileList, const QString &key, const T &defaultValue)
    {
        QList<T> result; result.reserve(fileList.size());
        foreach (const U &f, fileList) result.append(static_cast<const File&>(f).get<T>(key, defaultValue));
        return result;
    }

    QList<QPointF> namedPoints() const;
    QList<QPointF> points() const;
    void appendPoint(const QPointF &point);
    void appendPoints(const QList<QPointF> &points);
    inline void clearPoints() { m_metadata["Points"] = QList<QVariant>(); }
    inline void setPoints(const QList<QPointF> &points) { clearPoints(); appendPoints(points); }

    QList<QRectF> namedRects() const;
    QList<QRectF> rects() const;
    void appendRect(const QRectF &rect);
    void appendRect(const cv::Rect &rect);
    void appendRects(const QList<QRectF> &rects);
    void appendRects(const QList<cv::Rect> &rects);
    inline void clearRects() { m_metadata["Rects"] = QList<QVariant>(); }
    inline void setRects(const QList<QRectF> &rects) { clearRects(); appendRects(rects); }
    inline void setRects(const QList<cv::Rect> &rects) { clearRects(); appendRects(rects); }

    QList<cv::RotatedRect> namedRotatedRects() const;

    bool fte;
private:
    QVariantMap m_metadata;
    BR_EXPORT friend QDataStream &operator<<(QDataStream &stream, const File &file);
    BR_EXPORT friend QDataStream &operator>>(QDataStream &stream, File &file);

    void init(const QString &file);
};


template <>
inline bool File::get<bool>(const QString &key, const bool &defaultValue) const
{
    return getBool(key, defaultValue);
}


template <>
inline bool File::get<bool>(const QString &key) const
{
    return getBool(key);
}

BR_EXPORT QDebug operator<<(QDebug dbg, const File &file);
BR_EXPORT QDataStream &operator<<(QDataStream &stream, const File &file);
BR_EXPORT QDataStream &operator>>(QDataStream &stream, File &file);


struct BR_EXPORT FileList : public QList<File>
{
    FileList() {}
    FileList(int n);
    FileList(const QStringList &files);
    FileList(const QList<File> &files) { append(files); }

    QStringList flat() const;
    QStringList names() const;
    void sort(const QString& key);

    QList<int> crossValidationPartitions() const;
    int failures() const;

    static FileList fromGallery(const File &gallery, bool cache = false);
};


struct Template : public QList<cv::Mat>
{
    File file;
    Template() {}
    Template(const File &_file) : file(_file) {}
    Template(const File &_file, const cv::Mat &mat) : file(_file) { append(mat); }
    Template(const File &_file, const QList<cv::Mat> &mats) : file(_file) { append(mats); }
    Template(const cv::Mat &mat) { append(mat); }

    inline const cv::Mat &m() const { static const cv::Mat NullMatrix;
                                      return isEmpty() ? qFatal("Empty template."), NullMatrix : last(); }
    inline cv::Mat &m() { return isEmpty() ? append(cv::Mat()), last() : last(); }
    inline operator const File &() const { return file; }
    inline cv::Mat &operator=(const cv::Mat &other) { return m() = other; }
    inline operator const cv::Mat&() const { return m(); }
    inline operator cv::Mat&() { return m(); }
    inline operator cv::_InputArray() const { return m(); }
    inline operator cv::_OutputArray() { return m(); }
    inline bool isNull() const { return isEmpty() || !m().data; }
    inline void merge(const Template &other) { append(other); file.append(other.file); }

    size_t bytes() const
    {
        size_t bytes = 0;
        foreach (const cv::Mat &m, *this)
            bytes += m.total() * m.elemSize();
        return bytes;
    }

    Template clone() const
    {
        Template other(file);
        foreach (const cv::Mat &m, *this) other += m.clone();
        return other;
    }

    static br_utemplate toUniversalTemplate(const Template &t);
    static Template fromUniversalTemplate(br_const_utemplate ut);
    static br_utemplate readUniversalTemplate(QFile &file);
    static void writeUniversalTemplate(QFile &file, br_const_utemplate t);
    static void freeUniversalTemplate(br_const_utemplate t);
};

BR_EXPORT QDataStream &operator<<(QDataStream &stream, const Template &t);

BR_EXPORT QDataStream &operator>>(QDataStream &stream, Template &t);

struct TemplateList : public QList<Template>
{
    TemplateList() {}
    TemplateList(const QList<Template> &templates) { append(templates); }
    TemplateList(const QList<File> &files) { foreach (const File &file, files) append(file); }
    BR_EXPORT static TemplateList fromGallery(const File &gallery, bool partition = true);


    BR_EXPORT static TemplateList fromBuffer(const QByteArray &buffer);


    BR_EXPORT static TemplateList relabel(const TemplateList &tl, const QString &propName, bool preserveIntegers);

    /*!< \brief Assign templates to folds partitions. */
    BR_EXPORT TemplateList partition(const QString &inputVariable, bool random = false, bool overwrite = false) const;

    BR_EXPORT QList<int> indexProperty(const QString &propName, QHash<QString, int> * valueMap=NULL,QHash<int, QVariant> * reverseLookup = NULL) const;
    BR_EXPORT QList<int> indexProperty(const QString &propName, QHash<QString, int> &valueMap, QHash<int, QVariant> &reverseLookup) const;
    BR_EXPORT QList<int> applyIndex(const QString &propName, const QHash<QString, int> &valueMap) const;

    template <typename T>
    T bytes() const
    {
        T bytes = 0;
        foreach (const Template &t, *this) bytes += t.bytes();
        return bytes;
    }

    QList<cv::Mat> data(int index = 0) const
    {
        QList<cv::Mat> data; data.reserve(size());
        foreach (const Template &t, *this) data.append(t[index]);
        return data;
    }

    QList<TemplateList> split(const QList<int> &partitionSizes) const
    {
        int sum = 0;
        QList<TemplateList> partitions; partitions.reserve(partitionSizes.size());

        for (int i=0; i<partitionSizes.size(); i++) {
            partitions.append(TemplateList());
            sum+=partitionSizes[i];
        }

        if (sum != first().size()) qFatal("Partition sizes %i do not span template matrices %i properly", sum, first().size());

        foreach (const Template &t, *this) {
            int index = 0;
            while (index < t.size()) {
                for (int i=0; i<partitionSizes.size(); i++) {
                    Template newTemplate;
                    newTemplate.file = t.file;
                    for (int j=0; j<partitionSizes[i]; j++) {
                        newTemplate.append(t[index]);
                        index++;
                    }
                    // Append to the ith element of partitions
                    partitions[i].append(newTemplate);
                }
            }
        }

        return partitions;
    }

    FileList files() const
    {
        FileList files; files.reserve(size());
        foreach (const Template &t, *this) files.append(t.file);
        return files;
    }

    FileList operator()() const { return files(); }

    template<typename T>
    QMap<T,int> countValues(const QString &propName, bool excludeFailures = false) const
    {
        QMap<T, int> labelCounts;
        foreach (const File &file, files())
            if (!excludeFailures || !file.fte)
                labelCounts[file.get<T>(propName)]++;
        return labelCounts;
    }

    TemplateList reduced() const
    {
        Template reduced;
        foreach (const Template &t, *this)
            reduced.merge(t);
        return TemplateList() << reduced;
    }

    template<typename T>
    QList<int> find(const QString& key, const T& value)
    {
        QList<int> indices;
        for (int i=0; i<size(); i++)
            if (at(i).file.contains(key))
                if (at(i).file.get<T>(key) == value)
                    indices.append(i);
        return indices;
    }
};


class BR_EXPORT Object : public QObject
{
    Q_OBJECT
    int firstAvailablePropertyIdx;

public:
    File file;

    virtual void init() {}
    virtual void store(QDataStream &stream) const;
    virtual void load(QDataStream &stream);


    virtual void serialize(QDataStream &stream) const
    {
        stream << description();
        store(stream);
    }

    QStringList parameters() const;
    QStringList prunedArguments(bool expanded = false) const;
    QString argument(int index, bool expanded) const;
    virtual QString description(bool expanded = false) const;

    void setProperty(const QString &name, QVariant value);
    virtual bool setPropertyRecursive(const QString &name, QVariant value);
    bool setExistingProperty(const QString &name, QVariant value);

    virtual QList<Object *> getChildren() const;

    template<typename T>
    QList<T *> getChildren() const
    {
        QList<Object *> children = getChildren();
        QList<T *> output;
        foreach(Object *obj, children) {
            T *temp = dynamic_cast<T *>(obj);
            if (temp != NULL)
                output.append(temp);
        }
        return output;
    }

    static QStringList parse(const QString &string, char split = ',');

private:
    template <typename T> friend struct Factory;
    friend class Context;
    void init(const File &file);
};


class BR_EXPORT Context : public Object
{
    Q_OBJECT
    QFile logFile;

public:

    Q_PROPERTY(QString sdkPath READ get_sdkPath WRITE set_sdkPath RESET reset_sdkPath)
    BR_PROPERTY(QString, sdkPath, "")

    Q_PROPERTY(QString algorithm READ get_algorithm WRITE set_algorithm RESET reset_algorithm)
    BR_PROPERTY(QString, algorithm, "")

    Q_PROPERTY(QString log READ get_log WRITE set_log RESET reset_log)
    BR_PROPERTY(QString, log, "")

    Q_PROPERTY(QString path READ get_path WRITE set_path RESET reset_path)
    BR_PROPERTY(QString, path, "")

    Q_PROPERTY(int parallelism READ get_parallelism WRITE set_parallelism RESET reset_parallelism)
    BR_PROPERTY(int, parallelism, std::max(1, QThread::idealThreadCount()+1))

    Q_PROPERTY(bool useGui READ get_useGui WRITE set_useGui RESET reset_useGui)
    BR_PROPERTY(bool, useGui, true)

    Q_PROPERTY(int blockSize READ get_blockSize WRITE set_blockSize RESET reset_blockSize)
    BR_PROPERTY(int, blockSize, parallelism * ((sizeof(void*) == 4) ? 128 : 1024))

    Q_PROPERTY(bool quiet READ get_quiet WRITE set_quiet RESET reset_quiet)
    BR_PROPERTY(bool, quiet, false)

    Q_PROPERTY(bool verbose READ get_verbose WRITE set_verbose RESET reset_verbose)
    BR_PROPERTY(bool, verbose, false)

    Q_PROPERTY(QString mostRecentMessage READ get_mostRecentMessage WRITE set_mostRecentMessage RESET reset_mostRecentMessage)
    BR_PROPERTY(QString, mostRecentMessage, "")

    Q_PROPERTY(double currentStep READ get_currentStep WRITE set_currentStep RESET reset_currentStep)
    BR_PROPERTY(double, currentStep, 0)

    Q_PROPERTY(double currentProgress READ get_currentProgress WRITE set_currentProgress RESET reset_currentProgress)
    BR_PROPERTY(double, currentProgress, 0)

    Q_PROPERTY(double totalSteps READ get_totalSteps WRITE set_totalSteps RESET reset_totalSteps)
    BR_PROPERTY(double, totalSteps, 0)

    Q_PROPERTY(bool enrollAll READ get_enrollAll WRITE set_enrollAll RESET reset_enrollAll)
    BR_PROPERTY(bool, enrollAll, false)

    typedef QHash<QString,QStringList> Filters;
    Q_PROPERTY(Filters filters READ get_filters WRITE set_filters RESET reset_filters)
    BR_PROPERTY(Filters, filters, Filters())

    Q_PROPERTY(QByteArray buffer READ get_buffer WRITE set_buffer RESET reset_buffer)
    BR_PROPERTY(QByteArray, buffer, QByteArray())

    Q_PROPERTY(bool scoreNormalization READ get_scoreNormalization WRITE set_scoreNormalization RESET reset_scoreNormalization)
    BR_PROPERTY(bool, scoreNormalization, true)

    Q_PROPERTY(int crossValidate READ get_crossValidate WRITE set_crossValidate RESET reset_crossValidate)
    BR_PROPERTY(int, crossValidate, 0)

    Q_PROPERTY(QList<QString> modelSearch READ get_modelSearch WRITE set_modelSearch RESET reset_modelSearch)
    BR_PROPERTY(QList<QString>, modelSearch, QList<QString>() )

    QHash<QString,QString> abbreviations;
    QTime startTime;

    bool contains(const QString &name);
    void printStatus();
    float progress() const;
    void setProperty(const QString &key, const QString &value);
    int timeRemaining() const;

    static bool checkSDKPath(const QString &sdkPath);
    static void initialize(int &argc, char **argv, QString sdkPath = "", bool useGui = true);
    static void finalize();
    static QString about();
    static QString version();
    static QString scratchPath();
    static QStringList objects(const char *abstractions = ".*", const char *implementations = ".*", bool parameters = true);

private:
    static void messageHandler(QtMsgType type, const QMessageLogContext &context, const QString &msg);
};

BR_EXPORT extern Context *Globals;


template <class T>
struct Factory
{
    virtual ~Factory() {}

    static T *make(const File &file)
    {
        QString name = file.get<QString>("plugin", "");
        if (name.isEmpty()) name = file.suffix();
        if (!names().contains(name)) {
            if      (names().contains("Empty") && name.isEmpty()) name = "Empty";
            else if (names().contains("Default"))                 name = "Default";
            else    qFatal("%s registry does not contain object named: %s", qPrintable(baseClassName()), qPrintable(name));
        }
        T *object = registry->value(name)->_make();
        static_cast<Object*>(object)->init(file);
        return object;
    }

    static QList< QSharedPointer<T> > makeAll()
    {
        QList< QSharedPointer<T> > objects;
        foreach (const QString &name, names()) {
            objects.append(QSharedPointer<T>(registry->value(name)->_make()));
            objects.last()->init("");
        }
        return objects;
    }

    static QStringList names() { return registry ? registry->keys() : QStringList(); }

    static QString parameters(const QString &name)
    {
        if (!registry) return QString();
        QScopedPointer<T> object(make("." + name));
        return object->parameters().join(", ");
    }

protected:
    Factory(QString name)
    {
        if (!registry) registry = new QMap<QString,Factory<T>*>();

        const QString abstraction = baseClassName();
        if (name.endsWith(abstraction)) name = name.left(name.size()-abstraction.size());
        if (name.startsWith("br::")) name = name.right(name.size()-4);
        if (registry->contains(name)) qFatal("%s registry already contains object named: %s", qPrintable(abstraction), qPrintable(name));
        registry->insert(name, this);
    }

private:
    static QMap<QString,Factory<T>*> *registry;

    static QString baseClassName() { return QString(T::staticMetaObject.className()).remove("br::"); }
    virtual T *_make() const = 0;
};

template <class T> QMap<QString, Factory<T>*>* Factory<T>::registry = 0;

template <class _Abstraction, class _Implementation>
class FactoryInstance : public Factory<_Abstraction>
{
    FactoryInstance() : Factory<_Abstraction>(_Implementation::staticMetaObject.className()) {}
    _Abstraction *_make() const { return new _Implementation(); }
    static const FactoryInstance registerThis;
};
template <class _Abstraction, class _Implementation>
const FactoryInstance<_Abstraction,_Implementation> FactoryInstance<_Abstraction,_Implementation>::registerThis;

#define BR_REGISTER(ABSTRACTION,IMPLEMENTATION)       \
template class                                        \
br::FactoryInstance<br::ABSTRACTION, IMPLEMENTATION>;


class BR_EXPORT Initializer : public Object
{
    Q_OBJECT

public:
    virtual ~Initializer() {}
    virtual void initialize() const = 0;
    virtual void finalize() const {}
};


class BR_EXPORT Output : public Object
{
    Q_OBJECT

public:
    Q_PROPERTY(int blockRows READ get_blockRows WRITE set_blockRows RESET reset_blockRows STORED false)
    Q_PROPERTY(int blockCols READ get_blockCols WRITE set_blockCols RESET reset_blockCols STORED false)
    BR_PROPERTY(int, blockRows, -1)
    BR_PROPERTY(int, blockCols, -1)

    FileList targetFiles;
    FileList queryFiles;
    bool selfSimilar;

    virtual ~Output() {}
    virtual void initialize(const FileList &targetFiles, const FileList &queryFiles);
    virtual void setBlock(int rowBlock, int columnBlock);
    virtual void setRelative(float value, int i, int j);

    static Output *make(const File &file, const FileList &targetFiles, const FileList &queryFiles);

private:
    QSharedPointer<Output> next;
    QPoint offset;
    virtual void set(float value, int i, int j) = 0;
};


class BR_EXPORT MatrixOutput : public Output
{
    Q_OBJECT

public:
    cv::Mat data;

    static MatrixOutput *make(const FileList &targetFiles, const FileList &queryFiles);

protected:
    QString toString(int row, int column) const;

private:
    void initialize(const FileList &targetFiles, const FileList &queryFiles);
    void set(float value, int i, int j);
};

class BR_EXPORT Format : public Object
{
    Q_OBJECT

public:
    virtual ~Format() {}
    virtual Template read() const = 0;
    virtual void write(const Template &t) const = 0;
    static Template read(const QString &file);
    static void write(const QString &file, const Template &t);
};

class BR_EXPORT Gallery : public Object
{
    Q_OBJECT
public:
    Q_PROPERTY(int readBlockSize READ get_readBlockSize WRITE set_readBlockSize RESET reset_readBlockSize STORED false)
    BR_PROPERTY(int, readBlockSize, Globals->blockSize)

    virtual ~Gallery() {}
    TemplateList read();
    FileList files();
    virtual TemplateList readBlock(bool *done) = 0;
    void writeBlock(const TemplateList &templates);
    virtual void write(const Template &t) = 0;
    static Gallery *make(const File &file);
    void init();

    virtual qint64 totalSize() { return std::numeric_limits<qint64>::max(); }
    virtual qint64 position() { return 0; }

private:
    QSharedPointer<Gallery> next;
};


class TemplateEvent : public QObject
{
    Q_OBJECT

public:
    void pulseSignal(const Template &output) const
    {
        emit theSignal(output);
    }

signals:
    void theSignal(const Template &output) const;
};


class BR_EXPORT Transform : public Object
{
    Q_OBJECT

public:
    bool independent, trainable;

    virtual ~Transform() {}
    static Transform *make(QString str, QObject *parent);
    static QSharedPointer<Transform> fromAlgorithm(const QString &algorithm, bool preprocess=false);
    static QSharedPointer<Transform> fromComparison(const QString &algorithm);

    virtual Transform *clone() const;


    virtual void train(const TemplateList &data);
    virtual void train(const QList<TemplateList> &data);


    virtual void project(const Template &src, Template &dst) const = 0;
    virtual void project(const TemplateList &src, TemplateList &dst) const;

    virtual void projectUpdate(const Template &src, Template &dst)
    {
        project(src, dst);
    }

    virtual void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        project(src,dst);
    }


    void projectUpdate(Template &srcdst)
    {
        Template dst;
        projectUpdate(srcdst, dst);
        srcdst = dst;
    }

    void projectUpdate(TemplateList &srcdst)
    {
        TemplateList dst;
        projectUpdate(srcdst, dst);
        srcdst = dst;
    }

    virtual void finalize(TemplateList &output) { output = TemplateList(); }
    virtual bool timeVarying() const { return false; }

    inline Template operator()(const Template &src) const
    {
        Template dst;
        dst.file = src.file;
        project(src, dst);
        return dst;
    }

    inline TemplateList operator()(const TemplateList &src) const
    {
        TemplateList dst;
        project(src, dst);
        return dst;
    }

    virtual Transform *smartCopy(bool &newTransform) { newTransform=false; return this;}
    virtual Transform *smartCopy() {bool junk; return smartCopy(junk);}

    virtual TemplateEvent *getEvent(const QString &name);

    static Transform *deserialize(QDataStream &stream)
    {
        QString desc;
        stream >> desc;
        Transform *res = Transform::make(desc, NULL);
        res->load(stream);
        return res;
    }

    virtual Transform * simplify(bool &newTransform) { newTransform = false; return this; }
    virtual QByteArray likely(const QByteArray &indentation) const { (void) indentation; return "src"; }

protected:
    Transform(bool independent = true, bool trainable = true);
    inline Transform *make(const QString &description) { return make(description, this); }
};

inline Template &operator>>(Template &srcdst, const Transform &f)
{
    srcdst = f(srcdst);
    return srcdst;
}

inline TemplateList &operator>>(TemplateList &srcdst, const Transform &f)
{
    srcdst = f(srcdst);
    return srcdst;
}

inline QDataStream &operator<<(QDataStream &stream, const Transform &f)
{
    f.store(stream);
    return stream;
}

inline QDataStream &operator>>(QDataStream &stream, Transform &f)
{
    f.load(stream);
    return stream;
}


class BR_EXPORT Distance : public Object
{
    Q_OBJECT

public:
    virtual ~Distance() {}
    static Distance *make(QString str, QObject *parent);

    static QSharedPointer<Distance> fromAlgorithm(const QString &algorithm);
    virtual bool trainable() { return true; }
    virtual void train(const TemplateList &src) = 0;
    virtual void compare(const TemplateList &target, const TemplateList &query, Output *output) const;
    virtual QList<float> compare(const TemplateList &targets, const Template &query) const;
    virtual float compare(const Template &a, const Template &b) const;
    virtual float compare(const cv::Mat &a, const cv::Mat &b) const;
    virtual float compare(const uchar *a, const uchar *b, size_t size) const;

protected:
    inline Distance *make(const QString &description) { return make(description, this); }

private:
    virtual void compareBlock(const TemplateList &target, const TemplateList &query, Output *output, int targetOffset, int queryOffset) const;

    friend struct AlgorithmCore;
    virtual bool compare(const File &targetGallery, const File &queryGallery, const File &output) const
        { (void) targetGallery; (void) queryGallery; (void) output; return false; }
};

class BR_EXPORT Representation : public Object
{
    Q_OBJECT

public:
    virtual ~Representation() {}

    static Representation *make(QString str, QObject *parent); /*!< \brief Make a representation from a string. */

    virtual Template preprocess(const Template &src) const { return src; }
    virtual void train(const TemplateList &data) { (void)data; }
    virtual float evaluate(const Template &src, int idx) const = 0;
    // By convention passing an empty list evaluates all features in the representation
    virtual cv::Mat evaluate(const Template &src, const QList<int> &indices = QList<int>()) const = 0;

    virtual cv::Size windowSize(int *dx = NULL, int *dy = NULL) const = 0; // dx and dy should indicate the change to the original window size after preprocessing
    virtual int numChannels() const { return 1; }
    virtual int numFeatures() const = 0;
    virtual int maxCatCount() const = 0;
};

class BR_EXPORT Classifier : public Object
{
    Q_OBJECT

public:
    virtual ~Classifier() {}

    static Classifier *make(QString str, QObject *parent);

    virtual void train(const TemplateList &data) { (void)data; }
    virtual float classify(const Template &src, bool process = true, float *confidence = NULL) const = 0;

    // Slots for representations
    virtual Template preprocess(const Template &src) const { return src; }
    virtual cv::Size windowSize(int *dx = NULL, int *dy = NULL) const = 0;
    virtual int numFeatures() const { return 0; }
};


BR_EXPORT bool IsClassifier(const QString &algorithm);

BR_EXPORT void Train(const File &input, const File &model);

BR_EXPORT void Enroll(const File &input, const File &gallery = File());

BR_EXPORT void Enroll(TemplateList &tmpl);

BR_EXPORT void Project(const File &input, const File &output);

BR_EXPORT void Compare(const File &targetGallery, const File &queryGallery, const File &output);

BR_EXPORT void CompareTemplateLists(const TemplateList &target, const TemplateList &query, Output *output);

BR_EXPORT void PairwiseCompare(const File &targetGallery, const File &queryGallery, const File &output);

BR_EXPORT void Convert(const File &fileType, const File &inputFile, const File &outputFile);

BR_EXPORT void Cat(const QStringList &inputGalleries, const QString &outputGallery);

BR_EXPORT void Deduplicate(const File &inputGallery, const File &outputGallery, const QString &threshold);

BR_EXPORT Transform *wrapTransform(Transform *base, const QString &target);

BR_EXPORT Transform *pipeTransforms(QList<Transform *> &transforms);



} // namespace br

Q_DECLARE_METATYPE(cv::Mat)
Q_DECLARE_METATYPE(cv::RotatedRect)
Q_DECLARE_METATYPE(br::File)
Q_DECLARE_METATYPE(br::FileList)
Q_DECLARE_METATYPE(br::Template)
Q_DECLARE_METATYPE(br::TemplateList)
Q_DECLARE_METATYPE(br::Transform*)
Q_DECLARE_METATYPE(br::Distance*)
Q_DECLARE_METATYPE(br::Representation*)
Q_DECLARE_METATYPE(br::Classifier*)
Q_DECLARE_METATYPE(QList<int>)
Q_DECLARE_METATYPE(QList<float>)
Q_DECLARE_METATYPE(QList<br::Transform*>)
Q_DECLARE_METATYPE(QList<br::Distance*>)
Q_DECLARE_METATYPE(QList<br::Representation*>)
Q_DECLARE_METATYPE(QList<br::Classifier*>)

#endif // __cplusplus

#endif // BR_OPENBR_PLUGIN_H
