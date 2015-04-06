# File

A file path with associated metadata.

The File is one of two important data structures in OpenBR (the [Template](#template) is the other).
It is typically used to store the path to a file on disk with associated metadata.
The ability to associate a key/value metadata table with the file helps keep the API simple while providing customizable behavior.

When querying the value of a metadata key, the value will first try to be resolved against the file's private metadata table.
If the key does not exist in its local table then it will be resolved against the properties in the global Context.
By design file metadata may be set globally using Context::setProperty to operate on all files.

Files have a simple grammar that allow them to be converted to and from strings.
If a string ends with a **]** or **)** then the text within the final **[]** or **()** are parsed as comma separated metadata fields.
By convention, fields within **[]** are expected to have the format <tt>[key1=value1, key2=value2, ..., keyN=valueN]</tt> where order is irrelevant.
Fields within **()** are expected to have the format <tt>(value1, value2, ..., valueN)</tt> where order matters and the key context dependent.
The left hand side of the string not parsed in a manner described above is assigned to [name](#qstring-name).

Values are not necessarily stored as strings in the metadata table.
The system will attempt to infer and convert them to their "native" type.
The conversion logic is as follows:

1. If the value starts with **[** and ends with **]** then it is treated as a comma separated list and represented with [QVariantList][QVariantList]. Each value in the list is parsed recursively.
2. If the value starts with **(** and ends with **)** and contains four comma separated elements, each convertable to a floating point number, then it is represented with [QRectF][QRectF].
3. If the value starts with **(** and ends with **)** and contains two comma separated elements, each convertable to a floating point number, then it is represented with [QPointF][QPointF].
4. If the value is convertable to a floating point number then it is represented with <tt>float</tt>.
5. Otherwise, it is represented with [QString][QString].

Metadata keys fall into one of two categories:
* camelCaseKeys are inputs that specify how to process the file.
* Capitalized_Underscored_Keys are outputs computed from processing the file.

Below are some of the most commonly occurring standardized keys:

Key             | Value          | Description
---             | ----           | -----------
name            | QString        | Contents of [name](#name)
separator       | QString        | Separate [name](#name) into multiple files
Index           | int            | Index of a template in a template list
Confidence      | float          | Classification/Regression quality
FTE             | bool           | Failure to enroll
FTO             | bool           | Failure to open
*_X             | float          | Position
*_Y             | float          | Position
*_Width         | float          | Size
*_Height        | float          | Size
*_Radius        | float          | Size
Label           | QString        | Class label
Theta           | float          | Pose
Roll            | float          | Pose
Pitch           | float          | Pose
Yaw             | float          | Pose
Points          | QList<QPointF> | List of unnamed points
Rects           | QList<Rect>    | List of unnamed rects
Age             | float          | Age used for demographic filtering
Gender          | QString        | Subject gender
Train           | bool           | The data is for training, as opposed to enrollment
_*              | *              | Reserved for internal use

---

## Members

### [QString][QString] name

Path to a file on disk

### bool fte

Failed to enroll. If true this file failed to be processed somewhere in the template enrollment algorithm

### [QVariantMap][QVariantMap] m_metadata

Map for storing metadata. It is a [QString][QString], [QVariant][QVariant] key value pairing.

---

## Constructors

### File()

Default constructor. Sets [FTE](#bool-fte) to false.

### File(const [QString][QString] &file)

Initializes the file by calling the private function init.

### File(const [QString][QString] &file, const [QVariant][QVariant] &label)

Initializes the file by calling the private function init. Append label to the [metadata](#qvariantmap-m_metadata) using the key "Label".

### File(const char \*file)

Initializes the file with a c-style string.

### File(const [QVariantMap][QVariantMap] &metadata)

Sets [FTE](#bool-fte) to false and sets the [file metadata](#qvariantmap-m_metadata) to metadata.

---

## Static Functions


### static [QVariant][QVariant] parse([QString][QString] &value) const

Try to convert value to a [QPointF][QPointF], [QRectF][QRectF], int or float. If a conversion is possible it returns the converted value, otherwise it returns the unconverted string.

    QString point = "(1, 1)";
    QString rect = "(1, 1, 5, 5)";
    QString integer = "1";
    QString fp = "1.0";
    QString string = "Hello World";

    File::parse(point);   // returns QVariant(QPointF, QPointF(1, 1))
    File::parse(rect);    // returns QVariant(QRectF, QRectF(1, 1, 5x5))
    File::parse(integer); // returns QVariant(int, 1)
    File::parse(fp);      // returns QVariant(float, 1.0f)
    File::parse(string);  // returns QVariant(QString, "Hello World")

### static [QList][QList]&lt;[QVariant][QVariant]&gt; values(const [QList][QList]&lt;U&gt; &fileList, const [QString][QString] &key)

This function requires a type specification in place of U. Valid types are [File](#file) and [QString][QString]. Returns a list of the values of the key in each of the given files.

    File f1, f2;
    f1.set("Key1", QVariant::fromValue<float>(1));
    f1.set("Key2", QVariant::fromValue<float>(2));
    f2.set("Key1", QVariant::fromValue<float>(3));

    File::values<File>(QList<File>() << f1 << f2, "Key1"); // returns [QVariant(float, 1),
                                                           //          QVariant(float, 3)]

### static [QList][QList]&lt;T&gt; get(const [QList][QList]&lt;U&gt; &fileList, const [QString][QString] &key)

This function requires a type specification in place of T and U. Valid types for U are [File](#file) and [QString][QString]. T can be any type. Returns a list of the values of the key in each of the given files. If the key doesn't exist in any of the files or the value cannot be converted to type T an error is thrown.

    File f1, f2;
    f1.set("Key1", QVariant::fromValue<float>(1));
    f1.set("Key2", QVariant::fromValue<float>(2));
    f2.set("Key1", QVariant::fromValue<float>(3));

    File::get<float, File>(QList<File>() << f1 << f2, "Key1");  // returns [1., 3.]
    File::get<float, File>(QList<File>() << f1 << f2, "Key2");  // Error: Key doesn't exist in f2
    File::get<QRectF, File>(QList<File>() << f1 << f2, "Key1"); // Error: float is not convertable to QRectF

### static [QList][QList]&lt;T&gt; get(const [QList][QList]&lt;U&gt; &fileList, const [QString][QString] &key, const T &defaultValue)

This function requires a type specification in place of T and U. Valid types for U are [File](#file) and [QString][QString]. T can be any type. Returns a list of the values of the key in each of the given files. If the key doesn't exist in any of the files or the value cannot be converted to type T the given defaultValue is returned.

    File f1, f2;
    f1.set("Key1", QVariant::fromValue<float>(1));
    f1.set("Key2", QVariant::fromValue<float>(2));
    f2.set("Key1", QVariant::fromValue<float>(3));

    File::get<float, File>(QList<File>() << f1 << f2, "Key1");                       // returns [1., 3.]
    File::get<float, File>(QList<File>() << f1 << f2, "Key2", QList<float>() << 1);  // returns [1.]
    File::get<QRectF, File>(QList<File>() << f1 << f2, "Key1, QList<QRectF>()");     // returns []

### [QDebug][QDebug] operator <<([QDebug][QDebug] dbg, const [File](#file) &file)

Calls [flat](#qstring-flat-const) on the given file and that streams that file to stderr.

    File file("../path/to/pictures/picture.jpg");
    file.set("Key", QString("Value"));

    qDebug() << file; // "../path/to/pictures/picture.jpg[Key=Value]" streams to stderr

### [QDataStream][QDataStream] &operator <<([QDataStream][QDataStream] &stream, const [File](#file) &file)

Serialize a file to a data stream.

    void store(QDataStream &stream)
    {
        File file("../path/to/pictures/picture.jpg");
        file.set("Key", QString("Value"));

        stream << file; // "../path/to/pictures/picture.jpg[Key=Value]" serialized to the stream
    }

### [QDataStream][QDataStream] &operator >>([QDataStream][QDataStream] &stream, [File](#file) &file)

Deserialize a file from a data stream.

    void load(QDataStream &stream)
    {
        File in("../path/to/pictures/picture.jpg");
        in.set("Key", QString("Value"));

        stream << in; // "../path/to/pictures/picture.jpg[Key=Value]" serialized to the stream

        File out;
        stream >> out;

        out.name; // returns "../path/to/pictures/picture.jpg"
        out.flat(); // returns "../path/to/pictures/picture.jpg[Key=Value]"
    }

---

## Functions

### Operator [QString][QString]() const

returns [name](#qstring-name). Allows Files to be used as [QString][QString].

### [QString][QString] flat() const

Returns the [name](#qstring-name) and [metadata](#qvariantmap-m_metadata) as string.

    File file("../path/to/pictures/picture.jpg");
    file.set("Key1", QVariant::fromValue<float>(1));
    file.set("Key2", QVariant::fromValue<float>(2));

    file.flat(); // returns "../path/to/pictures/picture.jpg[Key1=1,Key2=2]"

### [QString][QString] hash() const

Returns a hash of the file.

    File file("../path/to/pictures/picture.jpg");
    file.set("Key1", QVariant::fromValue<float>(1));
    file.set("Key2", QVariant::fromValue<float>(2));

    file.hash(); // returns "kElVwY"

### [QStringList][QStringList] localKeys() const

Returns an immutable version of the local metadata keys gotten by calling [metadata](#metadata).keys().

    File file("../path/to/pictures/picture.jpg");
    file.set("Key1", QVariant::fromValue<float>(1));
    file.set("Key2", QVariant::fromValue<float>(2));

    file.localKeys(); // returns [Key1, Key2]

### [QVariantMap][QVariantMap] localMetadata() const

returns an immutable version of the local [metadata](#qvariantmap-m_metadata).

    File file("../path/to/pictures/picture.jpg");
    file.set("Key1", QVariant::fromValue<float>(1));
    file.set("Key2", QVariant::fromValue<float>(2));

    file.localMetadata(); // return QMap(("Key1", QVariant(float, 1)) ("Key2", QVariant(float, 2)))

### void append([QVariantMap][QVariantMap] &localMetadata)

Add new metadata fields to [metadata](#qvariantmap-m_metadata).

    File f();
    f.set("Key1", QVariant::fromValue<float>(1));

    QVariantMap map;
    map.insert("Key2", QVariant::fromValue<float>(2));
    map.insert("Key3", QVariant::fromValue<float>(3));

    f.append(map);
    f.flat(); // returns "[Key1=1, Key2=2, Key3=3]"

### void append(const [File](#file) &other)

Append another file using the **;** separator. The file names are combined with the separator in between them. The metadata fields are combined. An additional field describing the separator is appended to the metadata.

    File f1("../path/to/pictures/picture1.jpg");
    f1.set("Key1", QVariant::fromValue<float>(1));

    File f2("../path/to/pictures/picture2.jpg");
    f2.set("Key2", QVariant::fromValue<float>(2));
    f2.set("Key3", QVariant::fromValue<float>(3));

    f1.append(f2);
    f1.name; // return "../path/to/pictures/picture1.jpg;../path/to/pictures/picture2.jpg"
    f1.localKeys(); // returns "[Key1, Key2, Key3, separator]"


### [File](#file) &operator +=(const [QMap][QMap]&lt;[QString][QString], [QVariant][QVariant]&gt; &other)

Shortcut operator to call [append](#void-appendqvariantmap-localmetadata).

### [File](#file) &operator +=(const [File](#file) &other)

Shortcut operator to call [append](#void-appendconst-file-other).

### [QList][QList]&lt;[File](#file)&gt; split() const

Parse [name](#qstring-name) and split on the **;** separator. Each split file has the same [metadata](#qvariantmap-m_metadata) as the joined file.

    File f1("../path/to/pictures/picture1.jpg");
    f1.set("Key1", QVariant::fromValue<float>(1));

    f1.split(); // returns [../path/to/pictures/picture1.jpg[Key1=1]]

    File f2("../path/to/pictures/picture2.jpg");
    f2.set("Key2", QVariant::fromValue<float>(2));
    f2.set("Key3", QVariant::fromValue<float>(3));

    f1.append(f2);
    f1.split(); // returns [../path/to/pictures/picture1.jpg[Key1=1, Key2=2, Key3=3, separator=;],
                    //          ../path/to/pictures/picture2.jpg[Key1=1, Key2=2, Key3=3, separator=;]]

### [QList][QList]&lt;[File](#file)&gt; split(const [QString][QString] &separator) const

Split the file on the given separator. Each split file has the same [metadata](#qvariantmap-m_metadata) as the joined file.

    File f("../path/to/pictures/picture1.jpg,../path/to/pictures/picture2.jpg");
    f.set("Key1", QVariant::fromValue<float>(1));
    f.set("Key2", QVariant::fromValue<float>(2));

    f.split(","); // returns [../path/to/pictures/picture1.jpg[Key1=1, Key2=2],
                              ../path/to/pictures/picture2.jpg[Key1=1, Key2=2]]

### void setParameter(int index, const [QVariant][QVariant]&value)

Insert a keyless value into the [metadata](#qvariantmap-m_metadata).

    File f;
    f.set("Key1", QVariant::fromValue<float>(1));
    f.set("Key2", QVariant::fromValue<float>(2));

    f.setParameter(1, QVariant::fromValue<float>(3));
    f.setParameter(5, QVariant::fromValue<float>(4));

    f.flat(); // returns "[Key1=1, Key2=2, Arg1=3, Arg5=4]"

### bool operator ==(const char \*other) const

Compare [name](#qstring-name) to c-style string other.

    File f("picture.jpg");

    f == "picture.jpg";       // returns true
    f == "other_picture.jpg"; // returns false

### bool operator ==(const [File](#file) &other) const

Compare [name](#qstring-name) and [metadata](#qvariantmap-m_metadata) to another file name and metadata for equality.

    File f1("picture1.jpg");
    File f2("picture1.jpg");

    f1 == f2; // returns true

    f1.set("Key1", QVariant::fromValue<float>(1));
    f2.set("Key2", QVariant::fromValue<float>(2));

    f1 == f2; // returns false (metadata doesn't match)

### bool operator !=(const [File](#file) &other) const

Compare [name](#qstring-name) and [metadata](#qvariantmap-m_metadata) to another file name and metadata for inequality.

    File f1("picture1.jpg");
    File f2("picture1.jpg");

    f1 != f2; // returns false

    f1.set("Key1", QVariant::fromValue<float>(1));
    f2.set("Key2", QVariant::fromValue<float>(2));

    f1 != f2; // returns true (metadata doesn't match)

### bool operator <(const [File](#file) &other) const

Compare [name](#qstring-name) to a different file name.

### bool operator <=(const [File](#file) &other) const

Compare [name](#qstring-name) to a different file name.

### bool operator >(const [File](#file) &other) const

Compare [name](#qstring-name) to a different file name.

### bool operator >=(const [File](#file) &other) const

Compare [name](#qstring-name) to a different file name.

### bool isNull() const

Returns true if [name](#qstring-name) and [metadata](#qvariantmap-m_metadata) are empty and false otherwise.

    File f;
    f.isNull(); // returns true

    f.set("Key1", QVariant::fromValue<float>(1));
    f.isNull(); // returns false

### bool isTerminal() const

Returns true if [name](#qstring-name) equals "Terminal".

### bool exists() const

Returns true if the file at [name](#qstring-name) exists on disk.

### [QString][QString] fileName() const

Returns the file's base name and extension.

    File file("../path/to/pictures/picture.jpg");
    file.fileName(); // returns "picture.jpg"

### [QString][QString] baseName() const

Returns the file's base name.

    File file("../path/to/pictures/picture.jpg");
    file.baseName(); // returns "picture"

### [QString][QString] suffix() const

Returns the file's extension.

    File file("../path/to/pictures/picture.jpg");
    file.suffix(); // returns "jpg"

### [QString][QString] path() const

Return's the path of the file, excluding the name.

    File file("../path/to/pictures/picture.jpg");
    file.suffix(); // returns "../path/to/pictures"

### [QString][QString] resolved() const

Returns [name](#qstring-name). If name does not exist it prepends name with the path in Globals->path.

### bool contains(const [QString][QString] &key) const

Returns True if the key is in the [metadata](#qvariantmap-m_metadata) and False otherwise.

    File file;
    file.set("Key1", QVariant::fromValue<float>(1));

    file.contains("Key1"); // returns true
    file.contains("Key2"); // returns false

### bool contains(const [QStringList][QStringList] &keys) const

Returns True if all of the keys are in the [metadata](#qvariantmap-m_metadata) and False otherwise.

    File file;
    file.set("Key1", QVariant::fromValue<float>(1));
    file.set("Key2", QVariant::fromValue<float>(2));

    file.contains(QStringList("Key1")); // returns true
    file.contains(QStringList() << "Key1" << "Key2") // returns true
    file.contains(QStringList() << "Key1" << "Key3"); // returns false

### [QVariant][QVariant] value(const [QString][QString] &key) const

Returns the value associated with key in the [metadata](#qvariantmap-m_metadata).

    File file;
    file.set("Key1", QVariant::fromValue<float>(1));
    file.value("Key1"); // returns QVariant(float, 1)

### void set(const [QString][QString] &key, const [QVariant][QVariant] &value)

Insert or overwrite the [metadata](#qvariantmap-m_metadata) key with the given value.

    File f;
    f.flat(); // returns ""

    f.set("Key1", QVariant::fromValue<float>(1));
    f.flat(); // returns "[Key1=1]"

### void set(const [QString][QString] &key, const [QString][QString] &value)

Insert or overwrite the [metadata](#qvariantmap-m_metadata) key with the given value.

    File f;
    f.flat(); // returns ""

    f.set("Key1", QString("1"));
    f.flat(); // returns "[Key1=1]"

### void setList(const [QString][QString] &key, const [QList][QList]&lt;T&gt; &value)

This function requires a type specification in place of T. Insert or overwrite the [metadata](#qvariantmap-m_metadata) key with the value. The value will remain a list and should be queried with the function [getList](#qlistt-getlistconst-qstring-key-const).

    File file;

    QList<float> list = QList<float>() << 1 << 2 << 3;
    file.setList<float>("List", list);
    file.getList<float>("List"); // return [1., 2. 3.]

### void remove(const [QString][QString] &key)

Remove the key value pair associated with the given key from the [metadata](#metadata)

    File f;
    f.set("Key1", QVariant::fromValue<float>(1));
    f.set("Key2", QVariant::fromValue<float>(2));

    f.flat(); // returns "[Key1=1, Key2=2]"

    f.remove("Key1");
    f.flat(); // returns "[Key2=2]"

### T get(const [QString][QString] &key)

This function requires a type specification in place of T. Try and get the value associated with the given key in the [metadata](#qvariantmap-m_metadata). If the key does not exist or cannot be converted to the given type an error is thrown.

    File f;
    f.set("Key1", QVariant::fromValue<float>(1));

    f.get<float>("Key1");  // returns 1
    f.get<float>("Key2");  // Error: Key2 is not in the metadata
    f.get<QRectF>("Key1"); // Error: A float can't be converted to a QRectF

### T get(const [QString][QString] &key, const T &defaultValue)

This function requires a type specification in place of T. Try and get the value associated with the given key in the [metadata](#qvariantmap-m_metadata). If the key does not exist or cannot be converted to the given type the defaultValue is returned.

    File f;
    f.set("Key1", QVariant::fromValue<float>(1));

    f.get<float>("Key1", 5);  // returns 1
    f.get<float>("Key2", 5);  // returns 5
    f.get<QRectF>("Key1", QRectF(0, 0, 10, 10)); // returns QRectF(0, 0, 10x10)

### bool getBool(const [QString][QString] &key, bool defaultValue = false)

This is a specialization of [get](#t-getconst-qstring-key) for the boolean type. If the key is not in the [metadata](#qvariantmap-m_metadata) the defaultValue is returned. If the key is in the metadata but the value cannot be converted to a bool **true** is returned. If the key is found and the value can be converted to a bool the value is returned.

    File f;
    f.set("Key1", QVariant::fromValue<bool>(true));
    f.set("Key2", QVariant::fromValue<float>(10));

    f.getBool("Key1");       // returns true
    f.getBool("Key2")        // returns true (key found)
    f.getBool("Key3");       // returns false (default value)
    f.getBool("Key3", true); // returns true (default value)

### [QList][QList]&lt;T&gt; getList(const [QString][QString] &key) const

This function requires a type specification in place of T. Similar to [get](#t-getconst-qstring-key) only this returns a list. If the key is not found or the value cannot be converted into a [QList][QList]&lt;T&gt; an error is thrown.

    File file;

    QList<float> list = QList<float>() << 1 << 2 << 3;
    file.setList<float>("List", list);

    file.getList<float>("List");  // return [1., 2. 3.]
    file.getList<QRectF>("List"); // Error: float cannot be converted to QRectF
    file.getList<float>("Key");   // Error: key doesn't exist

### [QList][QList]&lt;T&gt; getList(const [QString][QString] &key, const [QList][QList]&lt;T&gt; &defaultValue) const

This function requires a type specification in place of T. Similar to [get](#t-getconst-qstring-key-const-t-defaultvalue) only this returns a list. If the key is not found or the value cannot be converted into a [QList][QList]&lt;T&gt; the supplied defaultValue is returned.

    File file;

    QList<float> list = QList<float>() << 1 << 2 << 3;
    file.setList<float>("List", list);

    file.getList<float>("List", QList<float>());                  // return [1., 2. 3.]
    file.getList<QRectF>("List", QList<QRectF>());                // return []
    file.getList<float>("Key", QList<float>() << 1 << 2 << 3);    // return [1., 2., 3.]

### [QList][QList]&lt;[QPointF][QPointF]&gt; namedPoints() const

Find all of the points that can be parsed from [metadata](#qvariantmap-m_metadata) keys and return them. Only values that are convertable to [QPointF][QPointF] are found. Values that can be converted to [QList][QList]>&lt;[QPointF][QPointF]&gt; are not included.

    File file;
    file.set("Key1", QVariant::fromValue<QPointF>(QPointF(1, 1)));
    file.set("Key2", QVariant::fromValue<QPointF>(QPointF(2, 2)));
    file.set("Points", QVariant::fromValue<QPointF>(QPointF(3, 3)))

    f.namedPoints(); // returns [QPointF(1, 1), QPointF(2, 2), QPointF(3, 3)]

    file.setPoints(QList<QPointF>() << QPointF(3, 3)); // changes metadata["Points"] to QList<QPointF>
    f.namedPoints(); // returns [QPointF(1, 1), QPointF(2, 2)]

### [QList][QList]&lt;[QPointf][QPointF]>&gt; points() const

Returns the list of points stored in [metadata](#qvariantmap-m_metadata)["Points"]. A list is expected and a single point not in a list will not be returned. Convenience functions [appendPoint](#void-appendpointconst-qpointf-point), [appendPoints](#void-appendpointsconst-qlistqpointf-points), [clearPoints](#void-clearpoints) and [setPoints](#void-setpointsconst-qlistqpointf-points) have been provided to manipulate the internal points list.

    File file;
    file.set("Points", QVariant::fromValue<QPointF>(QPointF(1, 1)));
    file.points(); // returns [] (point is not in a list)

    file.setPoints(QList<QPointF>() << QPointF(2, 2));
    file.points(); // returns [QPointF(2, 2)]

### void appendPoint(const [QPointF][QPointF] &point)

Add a point to the file's points list stored in [metadata](#qvariantmap-m_metadata)["Points"]

    File file;
    file.points(); // returns []

    file.appendPoint(QPointF(1, 1));
    file.points(); // returns [QPointF(1, 1)]

### void appendPoints(const [QList][QList]&lt;[QPointF][QPointF]&gt; &points)

Add a list of points to the file's points list stored in [metadata](#qvariantmap-m_metadata)["Points"]

    File file;
    file.points(); // returns []

    file.appendPoints(QList<QPointF>() << QPointF(1, 1) << QPointF(2, 2));
    file.points(); // returns [QPointF(1, 1), QPointF(2, 2)]

### void clearPoints()

Clear the list of points stored in [metadata](#qvariantmap-m_metadata)["Points"].

    File file;
    file.appendPoints(QList<QPointF>() << QPointF(1, 1) << QPointF(2, 2));
    file.points(); // returns [QPointF(1, 1), QPointF(2, 2)]

    file.clearPoints();
    file.points(); // returns []

### void setPoints(const [QList][QList]&lt;[QPointF][QPointF]&gt; &points)

Clears the points stored in [metadata](#qvariantmap-m_metadata) and replaces them with points.

    File file;
    file.appendPoints(QList<QPointF>() << QPointF(1, 1) << QPointF(2, 2));
    file.points(); // returns [QPointF(1, 1), QPointF(2, 2)]

    file.setPoints(QList<QPointF>() << QPointF(3, 3) << QPointF(4, 4));
    file.points(); // returns [QPointF(3, 3), QPointF(4, 4)]

### [QList][QList]&lt;[QRectF][QRectF]&gt; namedRects() const

Find all of the rects that can be parsed from [metadata](#qvariantmap-m_metadata) keys and return them. Only values that are convertable to [QRectF][QRectF] are found. Values that can be converted to [QList][QList]&lt;[QRectF][QRectF]&gt; are not included.

    File file;
    file.set("Key1", QVariant::fromValue<QRectF>(QRectF(1, 1, 5, 5)));
    file.set("Key2", QVariant::fromValue<QRectF>(QRectF(2, 2, 5, 5)));
    file.set("Rects", QVariant::fromValue<QRectF>(QRectF(3, 3, 5, 5)));

    f.namedRects(); // returns [QRectF(1, 1, 5x5), QRectF(2, 2, 5x5), QRectF(3, 3, 5x5)]

    file.setRects(QList<QRectF>() << QRectF(3, 3, 5x5)); // changes metadata["Rects"] to QList<QRectF>
    f.namedRects(); // returns [QRectF(1, 1, 5x5), QRectF(2, 2, 5x5)]

### [QList][QList]&lt;[QRectF][QRectF]&gt; rects() const

Returns the list of points stored in [metadata](#qvariantmap-m_metadata)["Rects"]. A list is expected and a single rect not in a list will not be returned. Convenience functions [appendRect](#void-appendrectconst-qrectf-rect), [appendRects](#void-appendrectsconst-qlistqrectf-rects), [clearRects](#void-clearrects) and [setRects](#void-setrectsconst-qlistqrectf-rects) have been provided to manipulate the internal points list.

    File file;
    file.set("Rects", QVariant::fromValue<QRectF>(QRectF(1, 1, 5, 5)));
    file.rects(); // returns [] (rect is not in a list)

    file.setRects(QList<QRectF>() << QRectF(2, 2, 5, 5));
    file.rects(); // returns [QRectF(2, 2, 5x5)]

### void appendRect(const [QRectF][QRectF] &rect)

Add a rect to the file's rects list stored in [metadata](#qvariantmap-m_metadata)["Rects"].

    File file;
    file.rects(); // returns []

    file.appendRect(QRectF(1, 1, 5, 5));
    file.rects(); // returns [QRectF(1, 1, 5x5)]

### void appendRect(const [Rect][Rect] &rect)

Add a OpenCV style rect to the file's rects list stored in [metadata](#qvariantmap-m_metadata)["Rects"]. The rect is automatically converted to a QRectF.

    File file;
    file.rects(); // returns []

    file.appendRect(cv::Rect(1, 1, 5, 5)); // automatically converted to QRectF
    file.rects(); // returns [QRectF(1, 1, 5x5)]

### void appendRects(const [QList][QList]&lt;[QRectF][QRectF]&gt; &rects)

Add a list of rects to the file's rects list stored in [metadata](#qvariantmap-m_metadata)["Rects"]

    File file;
    file.rects(); // returns []

    file.appendRects(QList<QRectF>() << QRectF(1, 1, 5, 5) << QRectF(2, 2, 5, 5));
    file.rects(); // returns [QRectF(1, 1, 5x5), QRectF(2, 2, 5x5)]

### void appendRects(const [QList][QList]&lt;[Rect][Rect]&gt; &rects)

Add a list of OpenCV style rects to the file's rects list stored in [metadata](#qvariantmap-m_metadata)["Rects"]. Each rect is automatically converted to a QRectF.

    File file;
    file.rects(); // returns []

    file.appendRects(QList<cv::Rect>() << cv::Rect(1, 1, 5, 5) << cv::Rect(2, 2, 5, 5));
    file.rects(); // returns [QRectF(1, 1, 5x5), QRectF(2, 2, 5x5)]


### void clearRects()

Clear the list of rects stored in [metadata](#qvariantmap-m_metadata)["Rects"].

    File file;
    file.appendRects(QList<QRectF>() << QRectF(1, 1, 5, 5) << QRectF(2, 2, 5, 5));
    file.rects(); // returns [QRectF(1, 1, 5x5), QRectF(2, 2, 5x5)]

    file.clearRects();
    file.rects(); // returns []

### void setRects(const [QList][QList]&lt;[QRectF][QRectF]&gt; &rects)

Clears the rects stored in [metadata](#qvariantmap-m_metadata)["Rects"] and replaces them with the given rects.

    File file;
    file.appendRects(QList<QRectF>() << QRectF(1, 1, 5, 5) << QRectF(2, 2, 5, 5));
    file.rects(); // returns [QRectF(1, 1, 5x5), QRectF(2, 2, 5x5)]

    file.setRects(QList<QRectF>() << QRectF(3, 3, 5, 5) << QRectF(4, 4, 5, 5));
    file.rects(); // returns [QRectF(3, 3, 5x5), QRectF(4, 4, 5x5)]

### void setRects(const [QList][QList]&lt;[Rect][Rect]&gt; &rects)

Clears the rects stored in [metadata](#qvariantmap-m_metadata)["Rects"] and replaces them with the given OpenCV style rects.

    File file;
    file.appendRects(QList<cv::Rect>() << cv::Rect(1, 1, 5, 5) << cv::Rect(2, 2, 5, 5));
    file.rects(); // returns [QRectF(1, 1, 5x5), QRectF(2, 2, 5x5)]

    file.setRects(QList<cv::Rect>() << cv::Rect(3, 3, 5, 5) << cv::Rect(4, 4, 5, 5));
    file.rects(); // returns [QRectF(3, 3, 5x5), QRectF(4, 4, 5x5)]

---

# FileList

Inherits [QList][QList]&lt;[File](#file)&gt;.

A convenience class for dealing with lists of files.

## Members

---

## Constructors

### FileList()

Default constructor. Doesn't do anything

### FileList(int n)

Initialize the [FileList](#filelist) with n empty files

### FileList(const [QStringList][QStringList] &files)

Initialize the [FileList](#filelist) from a list of strings. Each string should have the format "filename[key1=value1, key2=value2, ... keyN=valueN]"

### FileList(const [QList][QList]&lt;[File](#file)&gt; &files)

Initialize the [FileList](#filelist) from a list of [files](#file).

---

## Static Functions

### static FileList fromGallery(const [File](#file) &gallery, bool cache = false)

Creates a [FileList](#filelist) from a [Gallery](#gallery). Galleries store one or more [Templates](#template) on disk. Common formats include csv, xml, and gal, which is a unique OpenBR format. Read more about this in the [Gallery](#gallery) section. This function creates a [FileList](#filelist) by parsing the stored gallery based on its format. Cache determines whether the gallery should be stored for faster reading later.

    File gallery("gallery.csv");

    FileList fList = FileList::fromGallery(gallery);
    fList.flat(); // returns all the files that have been loaded from disk. It could
                  // be 1 or 100 depending on what was stored.

---

## Functions

### [QStringList][QStringList] flat() const

Calls [flat](#qstring-flat-const) on every [File](#file) in the list and returns the resulting strings as a [QStringList][QStringList].

    File f1("picture1.jpg"), f2("picture2.jpg");
    f1.set("Key", QString("Value"));

    FileList fList(QList<File>() << f1 << f2);
    fList.flat(); // returns ["picture1.jpg[Key=Value]", "picture2.jpg"]

### [QStringList][QStringList] names() const

Stores the name of every [file](#file) in the list and returns the resulting strings as a [QStringList][QStringList].

    File f1("picture1.jpg"), f2("picture2.jpg");
    f1.set("Key", QString("Value"));

    FileList fList(QList<File>() << f1 << f2);
    fList.names(); // returns ["picture1.jpg", "picture2.jpg"]

### void sort(const [QString][QString] &key)

Sorts the [FileList](#filelist) based on the value associated with the given key in each file.

    File f1("1"), f2("2"), f3("3");
    f1.set("Key", QVariant::fromValue<float>(3));
    f2.set("Key", QVariant::fromValue<float>(1));
    f3.set("Key", QVariant::fromValue<float>(2));

    FileList fList(QList<File>() << f1 << f2 << f3);
    fList.names(); // returns ["1", "2", "3"]

    fList.sort("Key");
    fList.names(); // returns ["2", "3", "1"]

### [QList][QList]&lt;int&gt; crossValidationPartitions() const

Returns the cross-validation partition (default=0) for each file in the list. The partition is stored with the [metadata](#qvariantmap-m_metadata) key "Partition".

    File f1, f2, f3;
    f1.set("Partition", QVariant::fromValue<int>(1));
    f3.set("Partition", QVariant::fromValue<int>(3));

    FileList fList(QList<File>() << f1 << f2 << f3);
    fList.crossValidationPartitions(); // returns [1, 0, 3]

### int failures() const

Returns the number of files that have [FTE](#bool-fte) = **True**.

    File f1, f2, f3;
    f1.fte = false;
    f2.fte = true;
    f3.fte = true;

    FileList fList(QList<File>() << f1 << f2 << f3);
    fList.failures(); // returns 2

---

# Template

Inherits [QList][QList]&lt;[Mat][Mat]&gt;.

A list of matrices associated with a file.

The Template is one of two important data structures in OpenBR (the [File](#file) is the other).
A template represents a biometric at various stages of enrollment and can be modified by [Transforms](#transform) and compared to other [templates](#template) with [Distance](#distance).

While there exist many cases (ex. video enrollment, multiple face detects, per-patch subspace learning, ...) where the template will contain more than one matrix,
in most cases templates have exactly one matrix in their list representing a single image at various stages of enrollment.
In the cases where exactly one image is expected, the template provides the function m() as an idiom for treating it as a single matrix.
Casting operators are also provided to pass the template into image processing functions expecting matrices.

Metadata related to the template that is computed during enrollment (ex. bounding boxes, eye locations, quality metrics, ...) should be assigned to the template's [File](#file-file) member.

## Members

### File file

The [File](#file) that constructs the [template](#template)

---

## Constructors

### Template()

The default template constructor. It doesn't do anything.

### Template(const [File](#file) &file)

Sets [file](#file-file) to the given file.

### Template(const [File](#file) &file, const [Mat][Mat] &mat)

Sets [file](#file-file) to the given file and appends the given mat to itself.

### Template(const [File](#file) &file, const [QList][QList]&lt;[Mat][Mat]&gt &mats)

Sets [file](#file-file) to the given file and appends the given mats to itself.

### Template(const [Mat][Mat] &mat)

Appends the given mat to itself.


---

## Static Functions

[QDataStream][QDataStream] &operator <<([QDataStream][QDataStream] &stream, const [Template](#template) &t)

Serialize a template.

    void store(QDataStream &stream)
    {
        Template t("picture.jpg");
        t.append(Mat::ones(1, 1, CV_8U));

        stream << t; // "["1"]picture.jpg" serialized to the stream
    }

[QDataStream][QDataStream] &operator >>([QDataStream][QDataStream] &stream, const [Template](#template) &t)

Deserialize a template.

    void load(QDataStream &stream)
    {
        Template in("picture.jpg");
        in.append(Mat::ones(1, 1, CV_8U));

        stream << in; // "["1"]picture.jpg" serialized to the stream

        Template out;
        stream >> out;

        out.file; // returns "picture.jpg"
        out; // returns ["1"]
    }

---

## Functions

### operator const [File](#file) &()

Idiom to treat the template like a [File](#file). Returns [file](#file-file).

### const [Mat][Mat] &m()

Idiom to treat the template like a [Mat][Mat]. If the template is empty then an empty [Mat][Mat] is returned. If the list has multiple [Mats][Mat] the last is returned.

    Template t;
    t.m(); // returns empty mat

    Mat m1;
    t.append(m1);
    t.m(); // returns m1;

    Mat m2;
    t.append(m2);
    t.m(); // returns m2;

### [Mat][Mat] &m()

Idiom to treat the template like a [Mat][Mat]. If the template is empty then an empty [Mat][Mat] is returned. If the list has multiple [Mats][Mat] the last is returned.

    Template t;
    t.m(); // returns empty mat

    Mat m1;
    t.append(m1);
    t.m(); // returns m1;

    Mat m2;
    t.append(m2);
    t.m(); // returns m2;

### operator const [Mat][Mat] &()

Idiom to treat the template like a [Mat][Mat]. Makes a call to [m()](#mat-m).

### operator [Mat][Mat] &()

Idiom to treat the template like a [Mat][Mat]. Makes a call to [m()](#mat-m).

### operator [_InputArray][InputArray] &()

Idiom to treat the template like a [Mat][Mat]. Makes a call to [m()](#mat-m).

### operator [_OutputArray][OutputArray] &()

Idiom to treat the template like a [Mat][Mat]. Makes a call to [m()](#mat-m).

### [Mat][Mat] &operator =(const [Mat][Mat] &other)

Idiom to treat the template like a [Mat][Mat]. Sets other equal to [m()](#mat-m).

### bool isNull() const

Returns true if the template is empty or if [m()](#mat-m) has no data.

    Template t;
    t.isNull(); // returns true

    t.append(Mat());
    t.isNull(); // returns true

    t.append(Mat::ones(1, 1, CV_8U));
    t.isNull(); // returns false

### void merge(const [Template](#template) &other)

Append the contents of another template. The [files](#file-file) are appended using [append()](#void-append-const-file-other).

    Template t1("picture1.jpg"), t2("picture2.jpg");
    Mat m1, m2;

    t1.append(m1);
    t2.append(m2);

    t1.merge(t2);

    t1.file; // returns picture1.jpg;picture2.jpg[seperator=;]
    t1; // returns [m1, m2]

### size_t bytes() const

Returns the total number of bytes in all of the matrices in the template.

    Template t;

    Mat m1 = Mat::ones(1, 1, CV_8U); // 1 byte
    Mat m2 = Mat::ones(2, 2, CV_8U); // 4 bytes
    Mat m3 = Mat::ones(3, 3, CV_8U); // 9 bytes

    t << m1 << m2 << m3;

    t.bytes(); // returns 14

### Template clone() const

Returns a new template with copies of the [file](#file-file) and all of the matrices that were in the original.

    Template t1("picture.jpg");
    t1.append(Mat::ones(1, 1, CV_8U));

    Template t2 = t1.clone();

    t2.file; // returns "picture.jpg"
    t2; // returns ["1"]

---

# TemplateList

Inherits [QList][QList]&lt;[Template][#template]&gt;.

Convenience class for working with a list of templates.

## Members

---

## Constructors

### TemplateList()

Default constructor.

### TemplateList(const [QList][QList]&lt;[Template](#template)&gt; &templates)

Initialize the [TemplateList](#templatelist) with a list of templates. The given list is appended.

### TemplateList(const [QList][QList]&lt;[File](#file)&gt; &files)

Initialize the [TemplateList](#templatelist) with a list of files. The files are each treated like a template and appended.

---

## Static Functions

### static [TemplateList](#templatelist) fromGallery(const [File](#file) &gallery)

Create a [TemplateList](#templatelist) from a [Gallery](#gallery).

### static [TemplateList](#templatelist) fromBuffer(const [QByteArray][QByteArray] &buffer)

Create a template from a memory buffer of individual templates. This is compatible with **.gal** galleries.

### static [TemplateList](#templatelist) relabel(const [TemplateList](#templatelist) &tl, const [QString][QString] &propName, bool preserveIntegers)

Relabels the metadata value associated with propName in each [Template](#template) to be between [0, numClasses-1]. **numClasses** is equal to the maximum value of the given metadata if the value can be converted to an **int** and **preserveIntegers** is true, or the total number of unique values. The relabeled values are stored in the "Label" field of each template.

    Template t1, t2, t3;

    t1.file.set("Class", QString("1"));
    t2.file.set("Class", QString("10"));
    t3.file.set("Class", QString("100"));
    TemplateList tList(QList<Template>() << t1 << t2 << t3);

    TemplateList relabeled = TemplateList::relabel(tList, "Class", true);
    relabeled.files(); // returns [[Class=1, Label=1], [Class=10, Label=10], [Class=100, Label=100]]

    relabeled = TemplateList::relabel(tList, "Class", false);
    relabeled.files(); // returns [[Class=1, Label=0], [Class=10, Label=1], [Class=100, Label=2]]

---

## Functions

### [QList][QList]&lt;int&gt; indexProperty(const [QString][QString] &propName, [QHash][QHash]&lt;[QString][QString], int&gt; &valueMap, [QHash][QHash]&lt;int, [QVariant][QVariant]&gt; &reverseLookup) const

Convert metadata values associated with **propName** to integers. Each unique value gets its own integer. Returns a list of the integer replacement for each template. This is useful in many classification problems where nominal data (e.g "Male", "Female") needs to represented with integers ("Male" = 0, "Female" = 1). **valueMap** and **reverseLookup** are created to allow easy conversion to the integer replacements and back.

    Template t1, t2, t3, t4;

    t1.file.set("Key", QString("Class 1"));
    t2.file.set("Key", QString("Class 2"));
    t3.file.set("Key", QString("Class 3"));
    t4.file.set("Key", QString("Class 1"));

    TemplateList tList(QList<Template>() << t1 << t2 << t3 << t4);

    QHash<QString, int> valueMap;
    QHash<int, QVariant> reverseLookup;
    tList.indexProperty("Key", valueMap, reverseLookup); // returns [0, 1, 2, 0]
    valueMap; // returns QHash(("Class 1", 0)("Class 2", 1)("Class 3", 2))
    reverseLookup; // QHash((0, QVariant(QString, "Class 1")) (2, QVariant(QString, "Class 3")) (1, QVariant(QString, "Class 2")))  


### [QList][QList]&lt;int&gt; indexProperty(const [QString][QString] &propName, [QHash][QHash]&lt;[QString][QString], int&gt; \*valueMap=NULL, [QHash][QHash]&lt;int, [QVariant][QVariant]&gt; \*reverseLookup=NULL) const

Shortcut to call [indexProperty](#qlistint-indexpropertyconst-qstring-propname-qhashqstring-int-valuemap-qhashint-qvariant-reverselookup-const) without **valueMap** or **reverseLookup** arguments.

### [QList][QList]&lt;int&gt; applyIndex(const [QString][QString] &propName, const [QHash][QHash]&lt;[QString][QString], int&gt; &valueMap) const

Apply a mapping to convert non-integer values to integers. Metadata values associated with **propName** are mapped through the given **valueMap**. If the value is found its integer mapping is appened to the list to be returned. If not, -1 is appened to the list.

    Template t1, t2, t3, t4;

    t1.file.set("Key", QString("Class 1"));
    t2.file.set("Key", QString("Class 2"));
    t3.file.set("Key", QString("Class 3"));
    t4.file.set("Key", QString("Class 1"));

    TemplateList tList(QList<Template>() << t1 << t2 << t3 << t4);

    QHash<QString, int> valueMap;
    valueMap.insert("Class 1", 0);
    valueMap.insert("Class 2", 1);

    tList.applyIndex("Key", valueMap); // returns [0, 1, -1, 0]

### T bytes() const

This function requires a type specification in place of T. Returns the total number of bytes in the [TemplateList](#templatelist). Calls [bytes()](#size_t-bytes-const) on each [Template](#template) in the list.

    Template t1, t2;

    t1.append(Mat::ones(1, 1, CV_8U)); // 1 byte
    t1.append(Mat::ones(2, 2, CV_8U)); // 4 bytes
    t2.append(Mat::ones(3, 3, CV_8U)); // 9 bytes
    t2.append(Mat::ones(4, 4, CV_8U)); // 16 bytes

    TemplateList tList(QList<Template>() << t1 << t2);
    tList.bytes(); // returns 30

### [QList][QList]&lt;[Mat][Mat]&gt; data(int index = 0) const

Returns a list of matrices. The list is compiled with one [Mat][Mat] from each [Template][#template] taken at the given index.

    Template t1, t2;

    t1.append(Mat::ones(1, 1, CV_8U));
    t1.append(Mat::zeros(1, 1, CV_8U));
    t2.append(Mat::ones(1, 1, CV_8U));
    t2.append(Mat::zeros(1, 1, CV_8U));

    TemplateList tList(QList<Template>() << t1 << t2);
    tList.data(); // returns ["1", "1"];
    tList.data(1); // returns ["0", "0"];

### [QList][QList]&lt;[TemplateList](#templatelist)&gt; partition(const [QList][QList]&lt;int&gt; &partitionSizes) const

Returns a [QList][QList] of [TemplateLists](#templatelist). The number of [TemplateLists](#templatelist) returned is equal to the length of partitionSizes. The number of [Templates](#template) in each returned  [TemplateList](#templatelist) is equal to the number of templates in the orginal [TemplateList](#templatelist). Each [Template](#template) in the original [TemplateList](#templatelist) must have length equal to the sum of the given partition sizes. Each [Template](#template) is divided into partitions and stored in the corresponding [TemplateList](#templatelist).

    Template t1, t2, t3;

    t1.append(Mat::ones(1, 1, CV_8U));
    t1.append(2*Mat::ones(1, 1, CV_8U));
    t1.append(3*Mat::ones(1, 1, CV_8U));

    t2.append(4*Mat::ones(1, 1, CV_8U));
    t2.append(5*Mat::ones(1, 1, CV_8U));
    t2.append(6*Mat::ones(1, 1, CV_8U));

    t3.append(7*Mat::ones(1, 1, CV_8U));
    t3.append(8*Mat::ones(1, 1, CV_8U));
    t3.append(9*Mat::ones(1, 1, CV_8U));

    TemplateList tList(QList<Template>() << t1 << t2 << t3);

    QList<TemplateList> partitions = tList.partition(QList<int>() << 1 << 2); // split into 2 partitions. 1 with 1 Mat and 1 with 2 Mats.

    partitions[0]; // returns [("1"), ("4"), ("7")]
    partitions[1]; // returns [("2", "3"), ("5", "6"), ("8", "9")]

### [FileList](#filelist) files() const

Returns a [FileList](#filelist) with the [file](#file-file) of each [Template](#template) in the [TemplateList](#templatelist).

    Template t1("picture1.jpg"), t2("picture2.jpg");

    t1.file.set("Key", QVariant::fromValue<float>(1));
    t2.file.set("Key", QVariant::fromValue<float>(2));

    TemplateList tList(QList<Template>() << t1 << t2);

    tList.files(); // returns ["picture1.jpg[Key=1]", "picture2.jpg[Key=2]"]

### [FileList](#filelist) operator()()

Returns [files()](#filelist-files-const).

### [QMap][QMap]&lt;T, int&gt; countValues(const [QString][QString] &propName, bool excludeFailures = false) const

This function requires a type specification in place of T. Returns the number of occurences for each label in the list. If excludedFailures is true [Files](#file) with [fte](#bool-fte) = true are excluded from the count.

    Template t1, t2, t3, t4;

    t1.file.set("Key", QString("Class 1"));
    t2.file.set("Key", QString("Class 2"));
    t3.file.set("Key", QString("Class 3"));
    t4.file.set("Key", QString("Class 1"));

    TemplateList tList(QList<Template>() << t1 << t2 << t3 << t4);

    tList.countValues<QString>("Key"); // returns QMap(("Class 1", 2), ("Class 2", 1), ("Class 3", 1))

### [TemplateList](#templatelist) reduced() const

Merge all of them templates together and store the resulting template in the list. The merges are done by calling [merge()](#void-merge-const-template-other).

    Template t1("picture1.jpg"), t2("picture2.jpg");

    t1.file.set("Key1", QString("Value1"));
    t2.file.set("Key2", QString("Value2"));

    TemplateList tList(QList<Template>() << t1 << t2);

    TemplateList reduced = tList.reduced();
    reduced.size(); // returns 1
    reduced.files(); // returns ["picture1.jpg;picture2.jpg[Key1=Value1, Key2=Value2, separator=;]"]

### [QList][QList]&lt;int&gt; find(const [QString][QString] &key, const T &value)

This function requires a type specification in place of T. Returns the indices of templates that have the given key-value pairing.

    Template t1, t2, t3;

    t1.file.set("Key", QString("Value1"));
    t2.file.set("Key", QString("Value2"));
    t3.file.set("Key", QString("Value2"));

    TemplateList tList(QList<Template>() << t1 << t2 << t3);
    tList.find<QString>("Key", "Value2"); // returns [1, 2]

---

# Context

---

# Object

---

# Factory

---

# Transform

---

# UntrainableTransform

---

# MetaTransform

---

# UntrainableMetaTransform

---

# MetadataTransform

---

# UntrainableMetadataTransform

---

# TimeVaryingTransform

---

# Distance

---

# UntrainableDistance

---

# Output

---

# MatrixOutput

---

# Format

---

# Representation

---

# Classifier



[QString]: http://doc.qt.io/qt-5/QString.html "QString"
[QStringList]: http://doc.qt.io/qt-5/qstringlist.html "QStringList"

[QList]: http://doc.qt.io/qt-5/QList.html "QList"
[QMap]: http://doc.qt.io/qt-5/qmap.html "QMap"
[QHash]: http://doc.qt.io/qt-5/qhash.html "QHash"

[QRectF]: http://doc.qt.io/qt-5/qrectf.html "QRectF"
[QPointF]: http://doc.qt.io/qt-5/qpointf.html "QPointF"

[QVariant]: http://doc.qt.io/qt-5/qvariant.html "QVariant"
[QVariantList]: http://doc.qt.io/qt-5/qvariant.html#QVariantList-typedef "QVariantList"
[QVariantMap]: http://doc.qt.io/qt-5/qvariant.html#QVariantMap-typedef "QVariantMap"

[QDebug]: http://doc.qt.io/qt-5/qdebug.html "QDebug"
[QDataStream]: http://doc.qt.io/qt-5/qdatastream.html "QDataStream"
[QByteArray]: http://doc.qt.io/qt-5/qbytearray.html "QByteArray"

[Mat]: http://docs.opencv.org/modules/core/doc/basic_structures.html#mat "Mat"
[Rect]: http://docs.opencv.org/modules/core/doc/basic_structures.html#rect "Rect"
[InputArray]: http://docs.opencv.org/modules/core/doc/basic_structures.html#inputarray "InputArray"
[OutputArray]: http://docs.opencv.org/modules/core/doc/basic_structures.html#outputarray "OutputArray"
