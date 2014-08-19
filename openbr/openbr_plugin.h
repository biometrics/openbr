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
#include <assert.h>

/*!
 * \defgroup cpp_plugin_sdk C++ Plugin SDK
 * \brief Plugin API for extending OpenBR functionality.
 *
 * \code
 * #include <openbr/openbr_plugin.h>
 * \endcode
 *
 * \par Development
 * Plugins should be developed in <tt>openbr/plugins/</tt>.
 * See this folder for numerous examples of existing plugins to follow when writing your own.
 * Plugins may optionally include a <tt>.cmake</tt> file to control build configuration.
 *
 * \par Documentation
 * Plugin documentation should include at least three lines providing the plugin abstraction, a brief explanation, and author.
 * If multiple authors are specified, the last author is assumed to be the current maintainer of the plugin.
 * Plugin authors are encouraged to <tt>\\cite</tt> relevant papers by adding them to <tt>share/openbr/openbr.bib</tt>.
 *
 * \section examples Examples
 * - \ref cpp_face_recognition
 * - \ref cpp_face_recognition_search
 * - \ref cpp_age_estimation
 * - \ref cpp_gender_estimation
 *
 * \subsection cpp_face_recognition Face Recognition
 * \ref cli_face_recognition "Command Line Interface Equivalent"
 * \snippet app/examples/face_recognition.cpp face_recognition
 *
 * \subsection cpp_face_recognition_search Face Recognition Search
 * \ref cli_face_recognition_search "Command Line Interface Equivalent"
 * \snippet app/examples/face_recognition_search.cpp face_recognition_search
 *
 * \subsection cpp_face_recognition_train Face Recognition Train
 * \ref cli_face_recognition_train "Command Line Interface Equivalent"
 * \snippet app/examples/face_recognition_train.cpp face_recognition_train
 *
 * \subsection cpp_age_estimation Age Estimation
 * \ref cli_age_estimation "Command Line Interface Equivalent"
 * \snippet app/examples/age_estimation.cpp age_estimation
 *
 * \subsection cpp_gender_estimation Gender Estimation
 * \ref cli_gender_estimation "Command Line Interface Equivalent"
 * \snippet app/examples/gender_estimation.cpp gender_estimation
 */

namespace br
{

/*!
 * \addtogroup cpp_plugin_sdk
 *  @{
 */

/*!
 * Helper macro for use with <a href="http://doc.qt.digia.com/qt/properties.html">Q_PROPERTY</a>.
 *
 * \b Example:<br>
 * Note the symmetry between \c BR_PROPERTY and \c Q_PROPERTY.
 * \snippet openbr/plugins/misc.cpp example_transform
 */
#define BR_PROPERTY(TYPE,NAME,DEFAULT)                  \
TYPE NAME;                                              \
TYPE get_##NAME() const { return NAME; }                \
void set_##NAME(TYPE the_##NAME) { NAME = the_##NAME; } \
void reset_##NAME() { NAME = DEFAULT; }

/*!
 * \brief A file path with associated metadata.
 *
 * The File is one of two important data structures in OpenBR (the Template is the other).
 * It is typically used to store the path to a file on disk with associated metadata.
 * The ability to associate a key/value metadata table with the file helps keep the API simple while providing customizable behavior.
 *
 * When querying the value of a metadata key, the value will first try to be resolved against the file's private metadata table.
 * If the key does not exist in its local table then it will be resolved against the properities in the global Context.
 * By design file metadata may be set globally using Context::setProperty to operate on all files.
 *
 * Files have a simple grammar that allow them to be converted to and from strings.
 * If a string ends with a \c ] or \c ) then the text within the final \c [] or \c () are parsed as comma sperated metadata fields.
 * By convention, fields within \c [] are expected to have the format <tt>[key1=value1, key2=value2, ..., keyN=valueN]</tt> where order is irrelevant.
 * Fields within \c () are expected to have the format <tt>(value1, value2, ..., valueN)</tt> where order matters and the key context dependent.
 * The left hand side of the string not parsed in a manner described above is assigned to #name.
 *
 * Values are not necessarily stored as strings in the metadata table.
 * The system will attempt to infer and convert them to their "native" type.
 * The conversion logic is as follows:
 * -# If the value starts with \c [ and ends with \c ] then it is treated as a comma separated list and represented with \c QVariantList. Each value in the list is parsed recursively.
 * -# If the value starts with \c ( and ends with \c ) and contains four comma separated elements, each convertable to a floating point number, then it is represented with \c QRectF.
 * -# If the value starts with \c ( and ends with \c ) and contains two comma separated elements, each convertable to a floating point number, then it is represented with \c QPointF.
 * -# If the value is convertable to a floating point number then it is represented with \c float.
 * -# Otherwise, it is represented with \c QString.
 *
 * Metadata keys fall into one of two categories:
 * - \c camelCaseKeys are inputs that specify how to process the file.
 * - \c Capitalized_Underscored_Keys are outputs computed from processing the file.
 *
 * Below are some of the most commonly occuring standardized keys:
 *
 * Key             | Value          | Description
 * ---             | ----           | -----------
 * name            | QString        | Contents of #name
 * separator       | QString        | Seperate #name into multiple files
 * Index           | int            | Index of a template in a template list
 * Confidence      | float          | Classification/Regression quality
 * FTE             | bool           | Failure to enroll
 * FTO             | bool           | Failure to open
 * *_X             | float          | Position
 * *_Y             | float          | Position
 * *_Width         | float          | Size
 * *_Height        | float          | Size
 * *_Radius        | float          | Size
 * Label           | QString        | Class label
 * Theta           | float          | Pose
 * Roll            | float          | Pose
 * Pitch           | float          | Pose
 * Yaw             | float          | Pose
 * Points          | QList<QPointF> | List of unnamed points
 * Rects           | QList<Rect>    | List of unnamed rects
 * Age             | float          | Age used for demographic filtering
 * Gender          | QString        | Subject gender
 * Train           | bool           | The data is for training, as opposed to enrollment
 * _*              | *              | Reserved for internal use
 */
struct BR_EXPORT File
{
    QString name; /*!< \brief Path to a file on disk. */

    File() { fte = false; }
    File(const QString &file) { init(file); } /*!< \brief Construct a file from a string. */
    File(const QString &file, const QVariant &label) { init(file); set("Label", label); } /*!< \brief Construct a file from a string and assign a label. */
    File(const char *file) { init(file); } /*!< \brief Construct a file from a c-style string. */
    File(const QVariantMap &metadata) : fte(false), m_metadata(metadata) {} /*!< \brief Construct a file from metadata. */
    inline operator QString() const { return name; } /*!< \brief Returns #name. */
    QString flat() const; /*!< \brief A stringified version of the file with metadata. */
    QString hash() const; /*!< \brief A hash of the file. */

    inline QStringList localKeys() const { return m_metadata.keys(); } /*!< \brief Returns the private metadata keys. */
    inline QVariantMap localMetadata() const { return m_metadata; } /*!< \brief Returns the private metadata. */

    void append(const QVariantMap &localMetadata); /*!< \brief Add new metadata fields. */
    void append(const File &other); /*!< \brief Append another file using \c separator. */
    inline File &operator+=(const QMap<QString,QVariant> &other) { append(other); return *this; } /*!< \brief Add new metadata fields. */
    inline File &operator+=(const File &other) { append(other); return *this; } /*!< \brief Append another file using \c separator. */

    QList<File> split() const; /*!< \brief Split the file using \c separator. */
    QList<File> split(const QString &separator) const; /*!< \brief Split the file. */

    inline void setParameter(int index, const QVariant &value) { set("_Arg" + QString::number(index), value); } /*!< \brief Insert a keyless value. */
    inline bool containsParameter(int index) const { return contains("_Arg" + QString::number(index)); } /*!< \brief Check for the existence of a keyless value. */
    inline QVariant getParameter(int index) const { return get<QVariant>("_Arg" + QString::number(index)); } /*!< \brief Retrieve a keyless value. */

    inline bool operator==(const char* other) const { return name == other; } /*!< \brief Compare name to c-style string. */
    inline bool operator==(const File &other) const { return (name == other.name) && (m_metadata == other.m_metadata); } /*!< \brief Compare name and metadata for equality. */
    inline bool operator!=(const File &other) const { return !(*this == other); } /*!< \brief Compare name and metadata for inequality. */
    inline bool operator<(const File &other) const { return name < other.name; } /*!< \brief Compare name. */
    inline bool operator<=(const File &other) const { return name <= other.name; } /*!< \brief Compare name. */
    inline bool operator>(const File &other) const { return name > other.name; } /*!< \brief Compare name. */
    inline bool operator>=(const File &other) const { return name >= other.name; } /*!< \brief Compare name. */

    inline bool isNull() const { return name.isEmpty() && m_metadata.isEmpty(); } /*!< \brief Returns \c true if name and metadata are empty, \c false otherwise. */
    inline bool isTerminal() const { return name == "terminal"; } /*!< \brief Returns \c true if #name is "terminal", \c false otherwise. */
    inline bool exists() const { return QFileInfo(name).exists(); } /*!< \brief Returns \c true if the file exists on disk, \c false otherwise. */
    inline QString fileName() const { return QFileInfo(name).fileName(); } /*!< \brief Returns the file's base name and extension. */
    inline QString baseName() const { const QString baseName = QFileInfo(name).baseName();
                                      return baseName.isEmpty() ? QDir(name).dirName() : baseName; } /*!< \brief Returns the file's base name. */
    inline QString suffix() const { return QFileInfo(name).suffix(); } /*!< \brief Returns the file's extension. */
    inline QString path() const { return QFileInfo(name).path(); } /*! \brief Returns the file's path excluding its name. */
    QString resolved() const; /*!< \brief Returns name prepended with Globals->path if name does not exist. */

    bool contains(const QString &key) const; /*!< \brief Returns \c true if the key has an associated value, \c false otherwise. */
    bool contains(const QStringList &keys) const; /*!< \brief Returns \c true if all keys have associated values, \c false otherwise. */
    QVariant value(const QString &key) const; /*!< \brief Returns the value for the specified key. */
    static QVariant parse(const QString &value); /*!< \brief Try to convert the QString to a QPointF or QRectF if possible. */
    inline void set(const QString &key, const QVariant &value) { m_metadata.insert(key, value); } /*!< \brief Insert or overwrite the metadata key with the specified value. */
    void set(const QString &key, const QString &value); /*!< \brief Insert or overwrite the metadata key with the specified value. */

    /*!< \brief Specialization for list type. Insert or overwrite the metadata key with the specified value. */
    template <typename T>
    void setList(const QString &key, const QList<T> &value)
    {
        QVariantList variantList;
        variantList.reserve(value.size());
        foreach (const T &item, value)
            variantList << item;
        set(key, variantList);
    }

    inline void remove(const QString &key) { m_metadata.remove(key); } /*!< \brief Remove the metadata key. */

    /*!< \brief Returns a value for the key, throwing an error if the key does not exist. */
    template <typename T>
    T get(const QString &key) const
    {
        if (!contains(key)) qFatal("Missing key: %s in: %s", qPrintable(key), qPrintable(flat()));
        QVariant variant = value(key);
        if (!variant.canConvert<T>()) qFatal("Can't convert: %s in: %s", qPrintable(key), qPrintable(flat()));
        return variant.value<T>();
    }

    /*!< \brief Returns a value for the key, returning \em defaultValue if the key does not exist or can't be converted. */
    template <typename T>
    T get(const QString &key, const T &defaultValue) const
    {
        if (!contains(key)) return defaultValue;
        QVariant variant = value(key);
        if (!variant.canConvert<T>()) return defaultValue;
        return variant.value<T>();
    }

    /*!< \brief Specialization for boolean type. */
    bool getBool(const QString &key, bool defaultValue = false) const;

    /*!< \brief Specialization for list type. Returns a list of type T for the key, throwing an error if the key does not exist or if the value cannot be converted to the specified type. */
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

    /*!< \brief Specialization for list type. Returns a list of type T for the key, returning \em defaultValue if the key does not exist or can't be converted. */
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

    /*!< \brief Returns the value for the specified key for every file in the list. */
    template<class U>
    static QList<QVariant> values(const QList<U> &fileList, const QString &key)
    {
        QList<QVariant> values; values.reserve(fileList.size());
        foreach (const U &f, fileList) values.append(((const File&)f).value(key));
        return values;
    }

    /*!< \brief Returns a value for the key for every file in the list, throwing an error if the key does not exist. */
    template<class T, class U>
    static QList<T> get(const QList<U> &fileList, const QString &key)
    {
        QList<T> result; result.reserve(fileList.size());
        foreach (const U &f, fileList) result.append(((const File&)f).get<T>(key));
        return result;
    }

    /*!< \brief Returns a value for the key for every file in the list, returning \em defaultValue if the key does not exist or can't be converted. */
    template<class T, class U>
    static QList<T> get(const QList<U> &fileList, const QString &key, const T &defaultValue)
    {
        QList<T> result; result.reserve(fileList.size());
        foreach (const U &f, fileList) result.append(static_cast<const File&>(f).get<T>(key, defaultValue));
        return result;
    }

    QList<QPointF> namedPoints() const; /*!< \brief Returns points convertible from metadata keys. */
    QList<QPointF> points() const; /*!< \brief Returns the file's points list. */
    void appendPoint(const QPointF &point); /*!< \brief Adds a point to the file's point list. */
    void appendPoints(const QList<QPointF> &points); /*!< \brief Adds landmarks to the file's landmark list. */
    inline void clearPoints() { m_metadata["Points"] = QList<QVariant>(); } /*!< \brief Clears the file's landmark list. */
    inline void setPoints(const QList<QPointF> &points) { clearPoints(); appendPoints(points); } /*!< \brief Overwrites the file's landmark list. */

    QList<QRectF> namedRects() const; /*!< \brief Returns rects convertible from metadata values. */
    QList<QRectF> rects() const; /*!< \brief Returns the file's rects list. */
    void appendRect(const QRectF &rect); /*!< \brief Adds a rect to the file's rect list. */
    void appendRect(const cv::Rect &rect); /*!< \brief Adds a rect to the file's rect list. */
    void appendRects(const QList<QRectF> &rects); /*!< \brief Adds rects to the file's rect list. */
    void appendRects(const QList<cv::Rect> &rects); /*!< \brief Adds rects to the file's rect list. */
    inline void clearRects() { m_metadata["Rects"] = QList<QVariant>(); } /*!< \brief Clears the file's rect list. */
    inline void setRects(const QList<QRectF> &rects) { clearRects(); appendRects(rects); } /*!< \brief Overwrites the file's rect list. */
    inline void setRects(const QList<cv::Rect> &rects) { clearRects(); appendRects(rects); } /*!< \brief Overwrites the file's rect list. */

    bool fte;
private:
    QVariantMap m_metadata;
    BR_EXPORT friend QDataStream &operator<<(QDataStream &stream, const File &file);
    BR_EXPORT friend QDataStream &operator>>(QDataStream &stream, File &file);

    void init(const QString &file);
};

/*!< \brief Specialization for boolean type. */
template <>
inline bool File::get<bool>(const QString &key, const bool &defaultValue) const
{
    return getBool(key, defaultValue);
}

/*!< \brief Specialization for boolean type. */
template <>
inline bool File::get<bool>(const QString &key) const
{
    return getBool(key);
}

BR_EXPORT QDebug operator<<(QDebug dbg, const File &file); /*!< \brief Prints br::File::flat() to \c stderr. */
BR_EXPORT QDataStream &operator<<(QDataStream &stream, const File &file); /*!< \brief Serializes the file to a stream. */
BR_EXPORT QDataStream &operator>>(QDataStream &stream, File &file); /*!< \brief Deserializes the file from a stream. */

/*!
 * \brief A list of files.
 *
 * Convenience class for working with a list of files.
 */
struct BR_EXPORT FileList : public QList<File>
{
    FileList() {}
    FileList(int n); /*!< \brief Initialize the list with \em n empty files. */
    FileList(const QStringList &files); /*!< \brief Initialize the file list from a string list. */
    FileList(const QList<File> &files) { append(files); } /*!< \brief Initialize the file list from another file list. */

    QStringList flat() const; /*!< \brief Returns br::File::flat() for each file in the list. */
    QStringList names() const; /*!<  \brief Returns #br::File::name for each file in the list. */
    void sort(const QString& key); /*!<  \brief Sort the list based on metadata. */

    QList<int> crossValidationPartitions() const; /*!< \brief Returns the cross-validation partition (default=0) for each file in the list. */
    int failures() const; /*!< \brief Returns the number of files with br::File::failed(). */

    static FileList fromGallery(const File &gallery, bool cache = false); /*!< \brief Create a file list from a br::Gallery. */
};

/*!
 * \brief A list of matrices associated with a file.
 *
 * The Template is one of two important data structures in OpenBR (the File is the other).
 * A template represents a biometric at various stages of enrollment and can be modified by br::Transform and compared to other templates with br::Distance.
 *
 * While there exist many cases (ex. video enrollment, multiple face detects, per-patch subspace learning, ...) where the template will contain more than one matrix,
 * in most cases templates have exactly one matrix in their list representing a single image at various stages of enrollment.
 * In the cases where exactly one image is expected, the template provides the function m() as an idiom for treating it as a single matrix.
 * Casting operators are also provided to pass the template into image processing functions expecting matrices.
 *
 * Metadata related to the template that is computed during enrollment (ex. bounding boxes, eye locations, quality metrics, ...) should be assigned to the template's #file member.
 */
struct Template : public QList<cv::Mat>
{
    File file; /*!< \brief The file from which the template is constructed. */
    Template() {}
    Template(const File &_file) : file(_file) {} /*!< \brief Initialize #file. */
    Template(const File &_file, const cv::Mat &mat) : file(_file) { append(mat); } /*!< \brief Initialize #file and append a matrix. */
    Template(const File &_file, const QList<cv::Mat> &mats) : file(_file) { append(mats); } /*!< \brief Initialize #file and append matricies. */
    Template(const cv::Mat &mat) { append(mat); } /*!< \brief Append a matrix. */

    inline const cv::Mat &m() const { static const cv::Mat NullMatrix;
                                      return isEmpty() ? qFatal("Empty template."), NullMatrix : last(); } /*!< \brief Idiom to treat the template as a matrix. */
    inline cv::Mat &m() { return isEmpty() ? append(cv::Mat()), last() : last(); } /*!< \brief Idiom to treat the template as a matrix. */
    inline operator const File &() const { return file; }
    inline cv::Mat &operator=(const cv::Mat &other) { return m() = other; } /*!< \brief Idiom to treat the template as a matrix. */
    inline operator const cv::Mat&() const { return m(); } /*!< \brief Idiom to treat the template as a matrix. */
    inline operator cv::Mat&() { return m(); } /*!< \brief Idiom to treat the template as a matrix. */
    inline operator cv::_InputArray() const { return m(); } /*!< \brief Idiom to treat the template as a matrix. */
    inline operator cv::_OutputArray() { return m(); } /*!< \brief Idiom to treat the template as a matrix. */
    inline bool isNull() const { return isEmpty() || !m().data; } /*!< \brief Returns \c true if the template is empty or has no matrix data, \c false otherwise. */
    inline void merge(const Template &other) { append(other); file.append(other.file); } /*!< \brief Append the contents of another template. */

    /*!
     * \brief Returns the total number of bytes in all the matrices.
     */
    size_t bytes() const
    {
        size_t bytes = 0;
        foreach (const cv::Mat &m, *this)
            bytes += m.total() * m.elemSize();
        return bytes;
    }

    /*!
     * \brief Copies all the matrices and returns a new template.
     */
    Template clone() const
    {
        Template other(file);
        foreach (const cv::Mat &m, *this) other += m.clone();
        return other;
    }
};

/*!
 * \brief Serialize a template.
 */
BR_EXPORT QDataStream &operator<<(QDataStream &stream, const Template &t);

/*!
 * \brief Deserialize a template.
 */
BR_EXPORT QDataStream &operator>>(QDataStream &stream, Template &t);

/*!
 * \brief A list of templates.
 *
 * Convenience class for working with a list of templates.
 */
struct TemplateList : public QList<Template>
{
    bool uniform; /*!< \brief Reserved for internal use. True if all templates are aligned and of the same size and type. */
    QVector<uchar> alignedData; /*!< \brief Reserved for internal use. */

    TemplateList() : uniform(false) {}
    TemplateList(const QList<Template> &templates) : uniform(false) { append(templates); } /*!< \brief Initialize the template list from another template list. */
    TemplateList(const QList<File> &files) : uniform(false) { foreach (const File &file, files) append(file); } /*!< \brief Initialize the template list from a file list. */
    BR_EXPORT static TemplateList fromGallery(const File &gallery); /*!< \brief Create a template list from a br::Gallery. */

    /*!< \brief Create a template list from a memory buffer of individual templates. Compatible with '.gal' galleries. */
    BR_EXPORT static TemplateList fromBuffer(const QByteArray &buffer);

    /*!< \brief Ensure labels are in the range [0,numClasses-1]. */
    BR_EXPORT static TemplateList relabel(const TemplateList &tl, const QString &propName, bool preserveIntegers);

    BR_EXPORT QList<int> indexProperty(const QString &propName, QHash<QString, int> * valueMap=NULL,QHash<int, QVariant> * reverseLookup = NULL) const;
    BR_EXPORT QList<int> indexProperty(const QString &propName, QHash<QString, int> &valueMap, QHash<int, QVariant> &reverseLookup) const;
    BR_EXPORT QList<int> applyIndex(const QString &propName, const QHash<QString, int> &valueMap) const;

    /*!
     * \brief Returns the total number of bytes in all the templates.
     */
    template <typename T>
    T bytes() const
    {
        T bytes = 0;
        foreach (const Template &t, *this) bytes += t.bytes();
        return bytes;
    }

    /*!
     * \brief Returns a list of matrices with one matrix from each template at the specified \em index.
     */
    QList<cv::Mat> data(int index = 0) const
    {
        QList<cv::Mat> data; data.reserve(size());
        foreach (const Template &t, *this) data.append(t[index]);
        return data;
    }

    /*!
     * \brief Returns a list of #br::TemplateList with each #br::Template in a given #br::TemplateList containing the number of matrices specified by \em partitionSizes.
     */
    QList<TemplateList> partition(const QList<int> &partitionSizes) const
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

    /*!
     * \brief Returns #br::Template::file for each template in the list.
     */
    FileList files() const
    {
        FileList files; files.reserve(size());
        foreach (const Template &t, *this) files.append(t.file);
        return files;
    }

    /*!
     * \brief Returns #br::Template::file for each template in the list.
     */
    FileList operator()() const { return files(); }

    /*!
     * \brief Returns the number of occurences for each label in the list.
     */
    template<typename T>
    QMap<T,int> countValues(const QString &propName, bool excludeFailures = false) const
    {
        QMap<T, int> labelCounts;
        foreach (const File &file, files())
            if (!excludeFailures || !file.fte)
                labelCounts[file.get<T>(propName)]++;
        return labelCounts;
    }

    /*!
     * \brief Merge all the templates together.
     */
    TemplateList reduced() const
    {
        Template reduced;
        foreach (const Template &t, *this)
            reduced.merge(t);
        return TemplateList() << reduced;
    }

    /*!
     * \brief Find the indices of templates with specified key, value pairs.
     */
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

/*!
 * \brief The base class of all plugins and objects requiring introspection.
 *
 * Plugins are constructed from files.
 * The file's name specifies which plugin to construct and the metadata provides initialization values for the plugin's properties.
 */
class BR_EXPORT Object : public QObject
{
    Q_OBJECT
    int firstAvailablePropertyIdx; /*!< \brief Index of the first property that can be set via command line arguments. */

public:
    File file; /*!< \brief The file used to construct the plugin. */

    virtual void init() {} /*!< \brief Overload this function instead of the default constructor to initialize the derived class. It should be safe to call this function multiple times. */
    virtual void store(QDataStream &stream) const; /*!< \brief Serialize the object. */
    virtual void load(QDataStream &stream); /*!< \brief Deserialize the object. Default implementation calls init() after deserialization. */

    /*!< \brief Serialize an object created via the plugin system, including the string used to build the base object, allowing re-creation of the object without knowledge of its base string*/
    virtual void serialize(QDataStream &stream) const
    {
        stream << description();
        store(stream);
    }

    QStringList parameters() const; /*!< \brief A string describing the parameters the object takes. */
    QStringList prunedArguments(bool expanded = false) const; /*!< \brief A string describing the values the object has, default valued parameters will not be listed. If expanded is true, all abbreviations and model file names should be replaced with a description of the object generated from those names. */
    QString argument(int index, bool expanded) const; /*!< \brief A string value for the argument at the specified index. */
    virtual QString description(bool expanded = false) const; /*!< \brief Returns a string description of the object. */
    
    void setProperty(const QString &name, QVariant value); /*!< \brief Overload of QObject::setProperty to handle OpenBR data types. */
    virtual bool setPropertyRecursive(const QString &name, QVariant value); /*!< \brief Recursive version of setProperty, try to set the property on this object, or its children, returns true if successful. */

    static QStringList parse(const QString &string, char split = ','); /*!< \brief Splits the string while respecting lexical scoping of <tt>()</tt>, <tt>[]</tt>, <tt>\<\></tt>, and <tt>{}</tt>. */

private:
    template <typename T> friend struct Factory;
    friend class Context;
    void init(const File &file); /*!< \brief Initializes the plugin's properties from the file's metadata. */
};

/*!
 * \brief The singleton class of global settings.
 *
 * Allocated by br::Context::initialize(), and deallocated by br::Context::finalize().
 *
 * \code
 * // Access the global context using the br::Globals pointer:
 * QString theSDKPath = br::Globals->sdkPath;
 * \endcode
 */
class BR_EXPORT Context : public Object
{
    Q_OBJECT
    QFile logFile;

public:
    /*!
     * \brief Path to <tt>share/openbr/openbr.bib</tt>
     */
    Q_PROPERTY(QString sdkPath READ get_sdkPath WRITE set_sdkPath RESET reset_sdkPath)
    BR_PROPERTY(QString, sdkPath, "")

    /*!
     * \brief The default algorithm to use when enrolling and comparing templates.
     */
    Q_PROPERTY(QString algorithm READ get_algorithm WRITE set_algorithm RESET reset_algorithm)
    BR_PROPERTY(QString, algorithm, "")

    /*!
     * \brief Optional log file to copy <tt>stderr</tt> to.
     */
    Q_PROPERTY(QString log READ get_log WRITE set_log RESET reset_log)
    BR_PROPERTY(QString, log, "")

    /*!
     * \brief Path to use when resolving images specified with relative paths.
     * Multiple paths can be specified using a semicolon separator.
     */
    Q_PROPERTY(QString path READ get_path WRITE set_path RESET reset_path)
    BR_PROPERTY(QString, path, "")

    /*!
     * \brief The number of threads to use.
     */
    Q_PROPERTY(int parallelism READ get_parallelism WRITE set_parallelism RESET reset_parallelism)
    BR_PROPERTY(int, parallelism, std::max(1, QThread::idealThreadCount()+1))

    /*!
     * \brief Whether or not to use GUI functions
     */
    Q_PROPERTY(bool useGui READ get_useGui WRITE set_useGui RESET reset_useGui)
    BR_PROPERTY(bool, useGui, true)

    /*!
     * \brief The maximum number of templates to process in parallel.
     */
    Q_PROPERTY(int blockSize READ get_blockSize WRITE set_blockSize RESET reset_blockSize)
    BR_PROPERTY(int, blockSize, parallelism * ((sizeof(void*) == 4) ? 128 : 1024))

    /*!
     * \brief If \c true no messages will be sent to the terminal, \c false by default.
     */
    Q_PROPERTY(bool quiet READ get_quiet WRITE set_quiet RESET reset_quiet)
    BR_PROPERTY(bool, quiet, false)

    /*!
     * \brief If \c true extra messages will be sent to the terminal, \c false by default.
     */
    Q_PROPERTY(bool verbose READ get_verbose WRITE set_verbose RESET reset_verbose)
    BR_PROPERTY(bool, verbose, false)

    /*!
     * \brief The most resent message sent to the terminal.
     */
    Q_PROPERTY(QString mostRecentMessage READ get_mostRecentMessage WRITE set_mostRecentMessage RESET reset_mostRecentMessage)
    BR_PROPERTY(QString, mostRecentMessage, "")

    /*!
     * \brief Used internally to compute progress() and timeRemaining().
     */

    Q_PROPERTY(double currentStep READ get_currentStep WRITE set_currentStep RESET reset_currentStep)
    BR_PROPERTY(double, currentStep, 0)

    Q_PROPERTY(double currentProgress READ get_currentProgress WRITE set_currentProgress RESET reset_currentProgress)
    BR_PROPERTY(double, currentProgress, 0)


    /*!
     * \brief Used internally to compute progress() and timeRemaining().
     */
    Q_PROPERTY(double totalSteps READ get_totalSteps WRITE set_totalSteps RESET reset_totalSteps)
    BR_PROPERTY(double, totalSteps, 0)

    /*!
     * \brief If \c true enroll 0 or more templates per image, otherwise (default) enroll exactly one.
     */
    Q_PROPERTY(bool enrollAll READ get_enrollAll WRITE set_enrollAll RESET reset_enrollAll)
    BR_PROPERTY(bool, enrollAll, false)

    typedef QHash<QString,QStringList> Filters;
    /*!
     * \brief Filters that automatically determine impostor matches based on target (gallery) template metadata.
     * \see br::FilterDistance
     */
    Q_PROPERTY(Filters filters READ get_filters WRITE set_filters RESET reset_filters)
    BR_PROPERTY(Filters, filters, Filters())

    /*!
     * \brief File output is redirected here if the file's basename is 'buffer', clearing previous contents.
     */
    Q_PROPERTY(QByteArray buffer READ get_buffer WRITE set_buffer RESET reset_buffer)
    BR_PROPERTY(QByteArray, buffer, QByteArray())

    /*!
     * \brief Enable/disable score normalization.
     */
    Q_PROPERTY(bool scoreNormalization READ get_scoreNormalization WRITE set_scoreNormalization RESET reset_scoreNormalization)
    BR_PROPERTY(bool, scoreNormalization, true)

    /*!
     * \brief Perform k-fold cross validation.
     */
    Q_PROPERTY(int crossValidate READ get_crossValidate WRITE set_crossValidate RESET reset_crossValidate)
    BR_PROPERTY(int, crossValidate, 0)

    QHash<QString,QString> abbreviations; /*!< \brief Used by br::Transform::make() to expand abbreviated algorithms into their complete definitions. */
    QTime startTime; /*!< \brief Used to estimate timeRemaining(). */

    /*!
     * \brief Returns the suggested number of partitions \em size should be divided into for processing.
     * \param size The length of the list to be partitioned.
     */
    int blocks(int size) const;

    /*!
     * \brief Returns true if \em name is queryable using <a href="http://doc.qt.digia.com/qt/qobject.html#property">QObject::property</a>
     * \param name The property key to check for existance.
     * \return \c true if \em name is a property, \c false otherwise.
     * \see set
     */
    bool contains(const QString &name);

    /*!
     * \brief Prints current progress statistics to \em stdout.
     * \see progress
     */
    void printStatus();

    /*!
     * \brief Returns the completion percentage of a call to br::Train(), br::Enroll() or br::Compare().
     * \return float Fraction completed.
     *  - \c -1 if no task is underway.
     * \see timeRemaining
     */
    float progress() const;

    /*!
     * \brief Set a global property.
     * \param key Global property key.
     * \param value Global property value.
     * \see contains
     */
    void setProperty(const QString &key, const QString &value);

    /*!
     * \brief Returns the time remaining in seconds of a call to \ref br_train, \ref br_enroll or \ref br_compare.
     * \return int Time remaining in seconds.
     *  - \c -1 if no task is underway.
     * \see progress
     */
    int timeRemaining() const;

    /*!
     * \brief Returns \c true if \em sdkPath is valid, \c false otherwise.
     * \param sdkPath The path to <tt>share/openbr/openbr.bib</tt>
     */
    static bool checkSDKPath(const QString &sdkPath);

    /*!
     * \brief Call \em once at the start of the application to allocate global variables.
     * \code
     * int main(int argc, char *argv[])
     * {
     *     br::Context::initialize(argc, argv);
     *
     *     // ...
     *
     *     br::Context::finalize();
     *     return 0;
     * }
     * \endcode
     * \param argc As provided by <tt>main()</tt>.
     * \param argv As provided by <tt>main()</tt>.
     * \param sdkPath The path to the folder containing <tt>share/openbr/openbr.bib</tt>
     *                 By default <tt>share/openbr/openbr.bib</tt> will be searched for relative to:
     *                   -# The working directory
     *                   -# The executable's location
     * \param useGui Create a QApplication instead of a QCoreApplication.
     * \note Tiggers \em abort() on failure to locate <tt>share/openbr/openbr.bib</tt>.
     * \note <a href="http://qt-project.org/">Qt</a> users should instead call this <i>after</i> initializing QApplication.
     * \see finalize
     */
    static void initialize(int &argc, char *argv[], QString sdkPath = "", bool useGui = true);

    /*!
     * \brief Call \em once at the end of the application to deallocate global variables.
     * \see initialize
     */
    static void finalize();

    /*!
     * \brief Returns a string with the name, version, and copyright of the project.
     * \return A string suitable for printing to the terminal or displaying in a dialog box.
     * \see version
     */
    static QString about();

    /*!
     * \brief Returns the version of the SDK.
     * \return A string with the format: <i>\<MajorVersion\></i><tt>.</tt><i>\<MinorVersion\></i><tt>.</tt><i>\<PatchVersion\></i>
     * \see about scratchPath
     */
    static QString version();

    /*!
     * \brief Returns the scratch directory.
     * \return A string with the format: <i>\</path/to/user/home/\></i><tt>OpenBR-</tt><i>\<MajorVersion\></i><tt>.</tt><i>\<MinorVersion\></i>
     * \note This should be used as the root directory for managing temporary files and providing process persistence.
     * \see version
     */
    static QString scratchPath();

    /*!
     * \brief Returns names and parameters for the requested objects.
     *
     * Each object is \c \\n seperated. Arguments are seperated from the object name with a \c \\t.
     * \param abstractions Regular expression of the abstractions to search.
     * \param implementations Regular expression of the implementations to search.
     * \param parameters Include parameters after object name.
     * \note \ref managed_return_value
     * \note This function uses Qt's <a href="http://doc.qt.digia.com/stable/qregexp.html">QRegExp</a> syntax.
     */
    static QStringList objects(const char *abstractions = ".*", const char *implementations = ".*", bool parameters = true);

private:
    static void messageHandler(QtMsgType type, const QMessageLogContext &context, const QString &msg);
};

/*!
 * \brief The globally available settings.
 *
 * Initialized by Context::initialize() and destroyed with Context::finalize().
 */
BR_EXPORT extern Context *Globals;

/*!
 * \brief For run time construction of objects from strings.
 *
 * All plugins must derive from br::Object.
 * The factory is a templated struct to allow for different types of plugins.
 *
 * Uses the Industrial Strength Pluggable Factory model described <a href="http://adtmag.com/articles/2000/09/25/industrial-strength-pluggable-factories.aspx">here</a>.
 */
template <class T>
struct Factory
{
    virtual ~Factory() {}

    /*!
     * \brief Constructs a plugin from a file.
     */
    //! [Factory make]
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
    //! [Factory make]

    /*!
     * \brief Constructs all the available plugins.
     */
    static QList< QSharedPointer<T> > makeAll()
    {
        QList< QSharedPointer<T> > objects;
        foreach (const QString &name, names()) {
            objects.append(QSharedPointer<T>(registry->value(name)->_make()));
            objects.last()->init("");
        }
        return objects;
    }

    /*!
     * \brief Returns the names of the available plugins.
     */
    static QStringList names() { return registry ? registry->keys() : QStringList(); }

    /*!
     * \brief Returns the parameters for a plugin.
     */
    static QString parameters(const QString &name)
    {
        if (!registry) return QString();
        QScopedPointer<T> object(registry->value(name)->_make());
        object->init(name);
        return object->parameters().join(", ");
    }

protected:
    /*!
     * \brief For internal use. Nifty trick to register objects using a constructor.
     */
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

/*!
 * Macro to register a br::Factory plugin.
 *
 * \b Example:<br>
 * Note the use of \c Q_OBJECT at the beginning of the class declaration and \c BR_REGISTER after the class declaration.
 * \snippet openbr/plugins/misc.cpp example_transform
 */
#define BR_REGISTER(ABSTRACTION,IMPLEMENTATION)       \
template class                                        \
br::FactoryInstance<br::ABSTRACTION, IMPLEMENTATION>;

/*!
 * \defgroup initializers Initializers
 * \brief Plugins that initialize resources.
 */

/*!
 * \ingroup initializers
 * \brief Plugin base class for initializing resources.
 */
class BR_EXPORT Initializer : public Object
{
    Q_OBJECT

public:
    virtual ~Initializer() {}
    virtual void initialize() const = 0;  /*!< \brief Called once at the end of br::Context::initialize(). */
    virtual void finalize() const {}  /*!< \brief Called once at the beginning of br::Context::finalize(). */
};

/*!
 * \defgroup outputs Outputs
 * \brief Plugins that store template comparison results.
 */

/*!
 * \ingroup outputs
 * \brief Plugin base class for storing template comparison results.
 *
 * An \em output is a br::File representing the result comparing templates.
 * br::File::suffix() is used to determine which plugin should handle the output.
 * \note Handle serialization to disk in the derived class destructor.
 */
class BR_EXPORT Output : public Object
{
    Q_OBJECT

public:
    Q_PROPERTY(int blockRows READ get_blockRows WRITE set_blockRows RESET reset_blockRows STORED false)
    Q_PROPERTY(int blockCols READ get_blockCols WRITE set_blockCols RESET reset_blockCols STORED false)
    BR_PROPERTY(int, blockRows, -1)
    BR_PROPERTY(int, blockCols, -1)

    FileList targetFiles; /*!< \brief List of files representing the gallery templates. */
    FileList queryFiles; /*!< \brief List of files representing the probe templates. */
    bool selfSimilar; /*!< \brief \c true if the \em targetFiles == \em queryFiles, \c false otherwise. */

    virtual ~Output() {}
    virtual void initialize(const FileList &targetFiles, const FileList &queryFiles); /*!< \brief Initializes class data members. */
    virtual void setBlock(int rowBlock, int columnBlock); /*!< \brief Set the current block. */
    virtual void setRelative(float value, int i, int j); /*!< \brief Set a score relative to the current block. */

    static Output *make(const File &file, const FileList &targetFiles, const FileList &queryFiles); /*!< \brief Make an output from a file and gallery/probe file lists. */

private:
    QSharedPointer<Output> next;
    QPoint offset;
    virtual void set(float value, int i, int j) = 0;
};

/*!
 * \ingroup outputs
 * \brief Plugin derived base class for storing outputs as matrices.
 */
class BR_EXPORT MatrixOutput : public Output
{
    Q_OBJECT

public:
    cv::Mat data; /*!< \brief The similarity matrix. */

    /*!
     * \brief Make a MatrixOutput from gallery and probe file lists.
     */
    static MatrixOutput *make(const FileList &targetFiles, const FileList &queryFiles);

protected:
    QString toString(int row, int column) const; /*!< \brief Converts the value requested similarity score to a string. */

private:
    void initialize(const FileList &targetFiles, const FileList &queryFiles);
    void set(float value, int i, int j);
};

/*!
 * \defgroup formats Formats
 * \brief Plugins that read a matrix.
 */

/*!
 * \ingroup formats
 * \brief Plugin base class for reading a template from disk.
 *
 * A \em format is a br::File representing a template (ex. jpg image) on disk.
 * br::File::suffix() is used to determine which plugin should handle the format.
 */
class BR_EXPORT Format : public Object
{
    Q_OBJECT

public:
    virtual ~Format() {}
    virtual Template read() const = 0; /*!< \brief Returns a br::Template created by reading #br::Object::file. */
    virtual void write(const Template &t) const = 0; /*!< \brief Writes the br::Template to #br::Object::file. */
};

/*!
 * \defgroup galleries Galleries
 * \brief Plugins that store templates.
 */

/*!
 * \ingroup galleries
 * \brief Plugin base class for storing a list of enrolled templates.
 *
 * A \em gallery is a file representing a br::TemplateList serialized to disk.
 * br::File::suffix() is used to determine which plugin should handle the gallery.
 * \note Handle serialization to disk in the derived class destructor.
 */
class BR_EXPORT Gallery : public Object
{
    Q_OBJECT    
public:
    Q_PROPERTY(int readBlockSize READ get_readBlockSize WRITE set_readBlockSize RESET reset_readBlockSize STORED false)
    BR_PROPERTY(int, readBlockSize, Globals->blockSize)

    virtual ~Gallery() {}
    TemplateList read(); /*!< \brief Retrieve all the stored templates. */
    FileList files(); /*!< \brief Retrieve all the stored template files. */
    virtual TemplateList readBlock(bool *done) = 0; /*!< \brief Retrieve a portion of the stored templates. */
    void writeBlock(const TemplateList &templates); /*!< \brief Serialize a template list. */
    virtual void write(const Template &t) = 0; /*!< \brief Serialize a template. */
    static Gallery *make(const File &file); /*!< \brief Make a gallery to/from a file on disk. */
    void init();

    virtual qint64 totalSize() { return std::numeric_limits<qint64>::max(); }
    virtual qint64 position() { return 0; }

private:
    QSharedPointer<Gallery> next;
};

/*!
 * \defgroup transforms Transforms
 * \brief Plugins that process a template.
 */

/*!
 * \addtogroup transforms
 *  @{
 */

/*!
 * \brief For asynchronous events during template projection.
 * \see Transform::getEvent
 */
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

/*!
 * \brief Plugin base class for processing a template.
 *
 * Transforms support the idea of \em training and \em projecting,
 * whereby they are (optionally) given example images and are expected learn how to transform new instances into an alternative,
 * hopefully more useful, basis for the recognition task at hand.
 * Transforms can be chained together to support the declaration and use of arbitrary algorithms at run time.
 */
class BR_EXPORT Transform : public Object
{
    Q_OBJECT

public:
    bool independent, trainable;

    virtual ~Transform() {}
    static Transform *make(QString str, QObject *parent); /*!< \brief Make a transform from a string. */
    static QSharedPointer<Transform> fromAlgorithm(const QString &algorithm, bool preprocess=true); /*!< \brief Retrieve an algorithm's transform. If preprocess is true, attaches a stream transform as the root of the algorithm*/
    static QSharedPointer<Transform> fromComparison(const QString &algorithm);

    virtual Transform *clone() const; /*!< \brief Copy the transform. */

    /*!< \brief Train the transform. */
    virtual void train(const TemplateList &data);

    /*!< \brief Train the transform, separate list items represent the way calls to project would be broken up
     * Transforms that have to call train on another transform should implement train(QList), the strucutre of the
     * list should mirror the calls that would be made to project by the parent transform. For example, DistributeTemplates
     * would make a separate project call for each template it receives, and therefore sets the QList to contain single item
     * template lists before passing it on.
     * This version of train(QList) is appropriate for transforms that perform training on themselves, and don't call train
     * on other transforms. It combines everything in data into a single TemplateList, then calls train(TemplateList)
     */
    virtual void train(const QList<TemplateList> &data);

    /*!< \brief Apply the transform to a single template. Typically used by independent transforms */
    virtual void project(const Template &src, Template &dst) const = 0;

    /*!< \brief Apply the transform, taking the full template list as input.
     * A TemplateList is what is typically passed from transform to transform. Transforms that just
     * need to operatoe on a single template at a time (and want to output exactly 1 template) can implement
     * project(template), but transforms that want to change the structure of the TemplateList (such as flatten), or
     * or output more or less than one template (e.g. detection methods) should implement project(TemplateList) directly
     */
    virtual void project(const TemplateList &src, TemplateList &dst) const;

    /*!< \brief Apply the transform to a single template, may update the transform's internal state
     * By default, just call project, we can always call a const function from a non-const function.
     * If a transform implements projectUpdate, it should report true to timeVarying so that it can be
     * handled correctly by e.g. Stream.
     */
    virtual void projectUpdate(const Template &src, Template &dst)
    {
        project(src, dst);
    }

    /*!< \brief Apply the transform, may update the transform's internal state. */
    virtual void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        project(src,dst);
    }

    /*!< \brief inplace projectUpdate. */
    void projectUpdate(Template &srcdst)
    {
        Template dst;
        projectUpdate(srcdst, dst);
        srcdst = dst;
    }

    /*!< \brief inplace projectUpdate. */
    void projectUpdate(TemplateList &srcdst)
    {
        TemplateList dst;
        projectUpdate(srcdst, dst);
        srcdst = dst;
    }

    /*!
     * Time-varying transforms may move away from a single input->single output model, and only emit
     * templates under some conditions (e.g. a tracking thing may emit a template for each detected
     * unique object), in this case finalize indicates that no further calls to project will be made
     * and the transform can emit a final set if templates if it wants. Time-invariant transforms
     * don't have to do anything.
     */
    virtual void finalize(TemplateList &output) { output = TemplateList(); }

    /*!
     * \brief Does the transform require the non-const version of project? Can vary for aggregation type transforms
     * (if their children are time varying, they are also time varying, otherwise probably not)
     */
    virtual bool timeVarying() const { return false; }

    /*!
     * \brief Convenience function equivalent to project().
     */
    inline Template operator()(const Template &src) const
    {
        Template dst;
        dst.file = src.file;
        project(src, dst);
        return dst;
    }

    /*!
     * \brief Convenience function equivalent to project().
     */
    inline TemplateList operator()(const TemplateList &src) const
    {
        TemplateList dst;
        project(src, dst);
        return dst;
    }

    /*!
     * \brief Perform the minimum amount of work necessary to make a
     * transform that can be used safely from a different thread than this
     * transform. For transforms that aren't time-varying, nothing needs to be
     * done, returning this is sufficient. Time varying transforms should implement this method
     * and copy enough of their state that projectUpdate can safely be called on the original
     * instance, and the copy concurrently.
     */
    virtual Transform *smartCopy(bool &newTransform) { newTransform=false; return this;}

    virtual Transform *smartCopy() {bool junk; return smartCopy(junk);}

    /*!
     * \brief Recursively retrieve a named event, returns NULL if an event is not found.
     */
    virtual TemplateEvent *getEvent(const QString &name);

    /*!
     * \brief Get a list of child transforms of this transform, child transforms are considered to be
     * any transforms stored as properties of this transform.
     */
    QList<Transform *> getChildren() const;

    static Transform *deserialize(QDataStream &stream)
    {
        QString desc;
        stream >> desc;
        Transform *res = Transform::make(desc, NULL);
        res->load(stream);
        return res;
    }

     /*!
     * \brief Return a pointer to a simplified version of this transform (if possible). Transforms which are only active during training should remove
     * themselves by either returning their child transforms (where relevant) or returning NULL. Set newTransform to true if the transform returned is newly allocated.
     */
    virtual Transform * simplify(bool &newTransform) { newTransform = false; return this; }

protected:
    Transform(bool independent = true, bool trainable = true); /*!< \brief Construct a transform. */
    inline Transform *make(const QString &description) { return make(description, this); } /*!< \brief Make a subtransform. */
};

/*!
 * \brief Convenience function equivalent to project().
 */
inline Template &operator>>(Template &srcdst, const Transform &f)
{
    srcdst = f(srcdst);
    return srcdst;
}

/*!
 * \brief Convenience function equivalent to project().
 */
inline TemplateList &operator>>(TemplateList &srcdst, const Transform &f)
{
    srcdst = f(srcdst);
    return srcdst;
}

/*!
 * \brief Convenience function equivalent to store().
 */
inline QDataStream &operator<<(QDataStream &stream, const Transform &f)
{
    f.store(stream);
    return stream;
}

/*!
 * \brief Convenience function equivalent to load().
 */
inline QDataStream &operator>>(QDataStream &stream, Transform &f)
{
    f.load(stream);
    return stream;
}
/*! @}*/

/*!
 * \defgroup distances Distances
 * \brief Plugins that compare templates.
 */

/*!
 * \ingroup distances
 * \brief Plugin base class for comparing templates.
 */
class BR_EXPORT Distance : public Object
{
    Q_OBJECT

public:
    virtual ~Distance() {}
    static Distance *make(QString str, QObject *parent); /*!< \brief Make a distance from a string. */

    static QSharedPointer<Distance> fromAlgorithm(const QString &algorithm); /*!< \brief Retrieve an algorithm's distance. */
    virtual void train(const TemplateList &src) { (void) src; } /*!< \brief Train the distance. */
    virtual void compare(const TemplateList &target, const TemplateList &query, Output *output) const; /*!< \brief Compare two template lists. */
    virtual QList<float> compare(const TemplateList &targets, const Template &query) const; /*!< \brief Compute the normalized distance between a template and a template list. */
    virtual float compare(const Template &a, const Template &b) const; /*!< \brief Compute the distance between two templates. */
    virtual float compare(const cv::Mat &a, const cv::Mat &b) const; /*!< \brief Compute the distance between two biometric signatures. */
    virtual float compare(const uchar *a, const uchar *b, size_t size) const; /*!< \brief Compute the distance between two buffers. */

protected:
    inline Distance *make(const QString &description) { return make(description, this); } /*!< \brief Make a subdistance. */

private:
    virtual void compareBlock(const TemplateList &target, const TemplateList &query, Output *output, int targetOffset, int queryOffset) const;

    friend struct AlgorithmCore;
    virtual bool compare(const File &targetGallery, const File &queryGallery, const File &output) const /*!< \brief Escape hatch for algorithms that need customized file I/O during comparison. */
        { (void) targetGallery; (void) queryGallery; (void) output; return false; }
};

/*!
* \brief Returns \c true if the algorithm is a classifier, \c false otherwise.
*
* Classifers have no br::Distance associated with their br::Transform.
* Instead they populate br::Template::file \c Label metadata field with the predicted class.
*/
BR_EXPORT bool IsClassifier(const QString &algorithm);

/*!
 * \brief High-level function for creating models.
 * \see br_train
 */
BR_EXPORT void Train(const File &input, const File &model);

/*!
 * \brief High-level function for creating galleries.
 * \see br_enroll
 */
BR_EXPORT void Enroll(const File &input, const File &gallery = File());

/*!
 * \brief High-level function for enrolling templates.
 * \see br_enroll
 */
BR_EXPORT void Enroll(TemplateList &tmpl);

/*!
 * \brief A naive alternative to \ref br::Enroll
 */
BR_EXPORT void Project(const File &input, const File &output);

/*!
 * \brief High-level function for comparing galleries.
 * \see br_compare
 */
BR_EXPORT void Compare(const File &targetGallery, const File &queryGallery, const File &output);
/*!
 * \brief High-level function for comparing templates.
 */
BR_EXPORT void CompareTemplateLists(const TemplateList &target, const TemplateList &query, Output *output);


/*!
 * \brief High-level function for doing a series of pairwise comparisons.
 * \see br_pairwise_compare
 */
BR_EXPORT void PairwiseCompare(const File &targetGallery, const File &queryGallery, const File &output);

/*!
 * \brief Change file formats.
 * \param fileType One of \c Format, \c Gallery, or \c Output.
 * \param inputFile The source file to convert from.
 * \param outputFile The destination file to convert to.
 */
BR_EXPORT void Convert(const File &fileType, const File &inputFile, const File &outputFile);

/*!
 * \brief Concatenate several galleries into one.
 * \param inputGalleries List of galleries to concatenate.
 * \param outputGallery Gallery to store the concatenated result.
 * \note outputGallery must not be in inputGalleries.
 */
BR_EXPORT void Cat(const QStringList &inputGalleries, const QString &outputGallery);

/*!
 * \brief Deduplicate a gallery.
 * \param inputGallery Gallery to deduplicate.
 * \param outputGallery Gallery to store the deduplicated result.
 * \param threshold Match score threshold to determine duplicates.
 */
BR_EXPORT void Deduplicate(const File &inputGallery, const File &outputGallery, const QString &threshold);

BR_EXPORT Transform *wrapTransform(Transform *base, const QString &target);

BR_EXPORT Transform *pipeTransforms(QList<Transform *> &transforms);

/*! @}*/

} // namespace br

Q_DECLARE_METATYPE(cv::Mat)
Q_DECLARE_METATYPE(br::File)
Q_DECLARE_METATYPE(br::FileList)
Q_DECLARE_METATYPE(br::Template)
Q_DECLARE_METATYPE(br::TemplateList)
Q_DECLARE_METATYPE(br::Transform*)
Q_DECLARE_METATYPE(br::Distance*)
Q_DECLARE_METATYPE(QList<int>)
Q_DECLARE_METATYPE(QList<float>)
Q_DECLARE_METATYPE(QList<br::Transform*>)
Q_DECLARE_METATYPE(QList<br::Distance*>)

#endif // __cplusplus

#endif // BR_OPENBR_PLUGIN_H
