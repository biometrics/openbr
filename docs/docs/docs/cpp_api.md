# Context

---

# File

A file path with associated metadata.

The File is one of two important data structures in OpenBR (the Template is the other).
It is typically used to store the path to a file on disk with associated metadata.
The ability to associate a key/value metadata table with the file helps keep the API simple while providing customizable behavior.

When querying the value of a metadata key, the value will first try to be resolved against the file's private metadata table.
If the key does not exist in its local table then it will be resolved against the properties in the global Context.
By design file metadata may be set globally using Context::setProperty to operate on all files.

Files have a simple grammar that allow them to be converted to and from strings.
If a string ends with a **]** or **)** then the text within the final **[]** or **()** are parsed as comma separated metadata fields.
By convention, fields within **[]** are expected to have the format <tt>[key1=value1, key2=value2, ..., keyN=valueN]</tt> where order is irrelevant.
Fields within **()** are expected to have the format <tt>(value1, value2, ..., valueN)</tt> where order matters and the key context dependent.
The left hand side of the string not parsed in a manner described above is assigned to #name.

Values are not necessarily stored as strings in the metadata table.
The system will attempt to infer and convert them to their "native" type.
The conversion logic is as follows:

1. If the value starts with **[** and ends with **]** then it is treated as a comma separated list and represented with [QVariantList](http://doc.qt.io/qt-5/qvariant.html#QVariantList-typedef). Each value in the list is parsed recursively.
2. If the value starts with **(** and ends with **)** and contains four comma separated elements, each convertable to a floating point number, then it is represented with [QRectF](http://doc.qt.io/qt-4.8/qrectf.html).
3. If the value starts with **(** and ends with **)** and contains two comma separated elements, each convertable to a floating point number, then it is represented with [QPointF](http://doc.qt.io/qt-4.8/qpointf.html).
4. If the value is convertable to a floating point number then it is represented with <tt>float</tt>.
5. Otherwise, it is represented with [QString](http://doc.qt.io/qt-5/QString.html).

Metadata keys fall into one of two categories:
* camelCaseKeys are inputs that specify how to process the file.
* Capitalized_Underscored_Keys are outputs computed from processing the file.

Below are some of the most commonly occuring standardized keys:

Key             | Value          | Description
---             | ----           | -----------
name            | QString        | Contents of #name
separator       | QString        | Seperate #name into multiple files
Index           | int            | Index of a template in a template list
Confidence      | float          | Classification/Regression quality
FTE             | bool           | Failure to enroll
FTO             | bool           | Failure to open
\*_X             | float          | Position
\*_Y             | float          | Position
\*_Width         | float          | Size
\*_Height        | float          | Size
\*_Radius        | float          | Size
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
_\*              | \*              | Reserved for internal use

---

## Members

### Name

Path to a file on disk

### FTE

Failed to enroll. If true this file failed to be processed somewhere in the template enrollment algorithm

### m_metadata

Map for storing metadata. It is a [QString](http://doc.qt.io/qt-5/QString.html), [QVariant](http://doc.qt.io/qt-5/qvariant.html) key value pairing.

---

## Constructors

### File()

Default constructor. Sets [FTE](#fte) to false.

### File(const [QString](http://doc.qt.io/qt-5/QString.html) &file)

Initializes the file by calling the private function init.

### File(const [QString](http://doc.qt.io/qt-5/QString.html) &file, const [QVariant](http://doc.qt.io/qt-5/qvariant.html) &label)

Initializes the file by calling the private function init. Append label to the [metadata](#m_metadata) using the key "Label".

### File(const char \*file)

Initializes the file with a c-style string.

### File(const [QVariantMap](http://doc.qt.io/qt-5/qvariant.html#QVariantMap-typedef) &metadata)

Sets [FTE](#file#fte) to false and sets the [file metadata](#m_metadata) to metadata.

---

## Static Functions


### static [QVariant](http://doc.qt.io/qt-5/qvariant.html) parse([QString](http://doc.qt.io/qt-5/QString.html) &value) const

Try to convert value to a [QPointF](http://doc.qt.io/qt-4.8/qpointf.html), [QRectF](http://doc.qt.io/qt-4.8/qrectf.html), int or float. If a conversion is possible it returns the converted value, otherwise it returns the unconverted string.

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

### static [QList](http://doc.qt.io/qt-5/QList.html)&lt;[QVariant](http://doc.qt.io/qt-5/qvariant.html)&gt; values(const [QList](http://doc.qt.io/qt-5/QList.html)&lt;U&gt; &fileList, const [QString](http://doc.qt.io/qt-5/QString.html) &key)

This function requires a type specification in place of U. Valid types are [File](#file) and [QString](http://doc.qt.io/qt-5/QString.html). Returns a list of the values of the key in each of the given files.

    File f1, f2;
    f1.set("Key1", QVariant::fromValue<float>(1));
    f1.set("Key2", QVariant::fromValue<float>(2));
    f2.set("Key1", QVariant::fromValue<float>(3));

    File::values<File>(QList<File>() << f1 << f2, "Key1"); // returns [QVariant(float, 1),
                                                           //          QVariant(float, 3)]

### static [QList](http://doc.qt.io/qt-5/QList.html)&lt;T&gt; get(const [QList](http://doc.qt.io/qt-5/QList.html)&lt;U&gt; &fileList, const [QString](http://doc.qt.io/qt-5/QString.html) &key)

This function requires a type specification in place of T and U. Valid types for U are [File](#file) and [QString](http://doc.qt.io/qt-5/QString.html). T can be any type. Returns a list of the values of the key in each of the given files. If the key doesn't exist in any of the files or the value cannot be converted to type T an error is thrown.

    File f1, f2;
    f1.set("Key1", QVariant::fromValue<float>(1));
    f1.set("Key2", QVariant::fromValue<float>(2));
    f2.set("Key1", QVariant::fromValue<float>(3));

    File::get<float, File>(QList<File>() << f1 << f2, "Key1");  // returns [1., 3.]
    File::get<float, File>(QList<File>() << f1 << f2, "Key2");  // Error: Key doesn't exist in f2
    File::get<QRectF, File>(QList<File>() << f1 << f2, "Key1"); // Error: float is not convertable to QRectF

### static [QList](http://doc.qt.io/qt-5/QList.html)&lt;T&gt; get(const [QList](http://doc.qt.io/qt-5/QList.html)&lt;U&gt; &fileList, const [QString](http://doc.qt.io/qt-5/QString.html) &key, const T &defaultValue)

This function requires a type specification in place of T and U. Valid types for U are [File](#file) and [QString](http://doc.qt.io/qt-5/QString.html). T can be any type. Returns a list of the values of the key in each of the given files. If the key doesn't exist in any of the files or the value cannot be converted to type T the given defaultValue is returned.

    File f1, f2;
    f1.set("Key1", QVariant::fromValue<float>(1));
    f1.set("Key2", QVariant::fromValue<float>(2));
    f2.set("Key1", QVariant::fromValue<float>(3));

    File::get<float, File>(QList<File>() << f1 << f2, "Key1");                       // returns [1., 3.]
    File::get<float, File>(QList<File>() << f1 << f2, "Key2", QList<float>() << 1);  // returns [1.]
    File::get<QRectF, File>(QList<File>() << f1 << f2, "Key1, QList<QRectF>()");     // returns []

### [QDebug](http://doc.qt.io/qt-5/qdebug.html) operator <<([QDebug](http://doc.qt.io/qt-5/qdebug.html) dbg, const [File](#file) &file)

Calls [flat](#qstring-flat-const) on the given file and that streams that file to stderr.

    File file("../path/to/pictures/picture.jpg");
    file.set("Key", QString("Value"));

    qDebug() << file; // "../path/to/pictures/picture.jpg[Key=Value]" streams to stderr

### [QDataStream](http://doc.qt.io/qt-5/qdatastream.html) &operator <<([QDataStream](http://doc.qt.io/qt-5/qdatastream.html) &stream, const [File](#file) &file)

Serialize a file to a data stream.

    void store(QDataStream &stream)
    {
        File file("../path/to/pictures/picture.jpg");
        file.set("Key", QString("Value"));

        stream << file; // "../path/to/pictures/picture.jpg[Key=Value]" serialized to the stream
    }

### [QDataStream](http://doc.qt.io/qt-5/qdatastream.html) &operator >>([QDataStream](http://doc.qt.io/qt-5/qdatastream.html) &stream, [File](#file) &file)

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

### Operator [QString](http://doc.qt.io/qt-5/QString.html)() const

returns [name](#name). Allows Files to be used as [QString](http://doc.qt.io/qt-5/QString.html).

### [QString](http://doc.qt.io/qt-5/QString.html) flat() const

Returns the [name](#name) and [metadata](#m_metadata) as string.

    File file("../path/to/pictures/picture.jpg");
    file.set("Key1", QVariant::fromValue<float>(1));
    file.set("Key2", QVariant::fromValue<float>(2));

    file.flat(); // returns "../path/to/pictures/picture.jpg[Key1=1,Key2=2]"

### [QString](http://doc.qt.io/qt-5/QString.html) hash() const

Returns a hash of the file.

    File file("../path/to/pictures/picture.jpg");
    file.set("Key1", QVariant::fromValue<float>(1));
    file.set("Key2", QVariant::fromValue<float>(2));

    file.hash(); // returns "kElVwY"

### [QStringList](http://doc.qt.io/qt-5/qstringlist.html) localKeys() const

Returns an immutable version of the local metadata keys gotten by calling [metadata](#metadata).keys().

    File file("../path/to/pictures/picture.jpg");
    file.set("Key1", QVariant::fromValue<float>(1));
    file.set("Key2", QVariant::fromValue<float>(2));

    file.localKeys(); // returns [Key1, Key2]

### [QVariantMap](http://doc.qt.io/qt-5/qvariant.html#QVariantMap-typedef) localMetadata() const

returns an immutable version of the local [metadata](#m_metadata).

    File file("../path/to/pictures/picture.jpg");
    file.set("Key1", QVariant::fromValue<float>(1));
    file.set("Key2", QVariant::fromValue<float>(2));

    file.localMetadata(); // return QMap(("Key1", QVariant(float, 1)) ("Key2", QVariant(float, 2)))

### void append([QVariantMap](http://doc.qt.io/qt-5/qvariant.html#QVariantMap-typedef) &localMetadata)

Add new metadata fields to [metadata](#m_metadata).

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


### File &operator +=(const [QMap](http://doc.qt.io/qt-5/qmap.html)&lt;[QString](http://doc.qt.io/qt-5/QString.html), [QVariant](http://doc.qt.io/qt-5/qvariant.html)&gt; &other)

Shortcut operator to call [append](#void-appendqvariantmap-localmetadata).

### File &operator +=(const [File](#file) &other)

Shortcut operator to call [append](#void-appendconst-file-other).

### [QList](http://doc.qt.io/qt-5/qlist.html)&lt;[File](#file)&gt; split() const

Parse [name](#name) and split on the **;** separator. Each split file has the same [metadata](#m_metadata) as the joined file.

    File f1("../path/to/pictures/picture1.jpg");
    f1.set("Key1", QVariant::fromValue<float>(1));

    f1.split(); // returns [../path/to/pictures/picture1.jpg[Key1=1]]

    File f2("../path/to/pictures/picture2.jpg");
    f2.set("Key2", QVariant::fromValue<float>(2));
    f2.set("Key3", QVariant::fromValue<float>(3));

    f1.append(f2);
    f1.split(); // returns [../path/to/pictures/picture1.jpg[Key1=1, Key2=2, Key3=3, separator=;],
                    //          ../path/to/pictures/picture2.jpg[Key1=1, Key2=2, Key3=3, separator=;]]

### [QList](http://doc.qt.io/qt-5/qlist.html)&lt;[File](#file)&gt; split(const [QString](http://doc.qt.io/qt-5/QString.html) &separator) const

Split the file on the given separator. Each split file has the same [metadata](#m_metadata) as the joined file.

    File f("../path/to/pictures/picture1.jpg,../path/to/pictures/picture2.jpg");
    f.set("Key1", QVariant::fromValue<float>(1));
    f.set("Key2", QVariant::fromValue<float>(2));

    f.split(","); // returns [../path/to/pictures/picture1.jpg[Key1=1, Key2=2],
                              ../path/to/pictures/picture2.jpg[Key1=1, Key2=2]]

### void setParameter(int index, const [QVariant](http://doc.qt.io/qt-5/qvariant.html) &value)

Insert a keyless value into the [metadata](#m_metadata).

    File f;
    f.set("Key1", QVariant::fromValue<float>(1));
    f.set("Key2", QVariant::fromValue<float>(2));

    f.setParameter(1, QVariant::fromValue<float>(3));
    f.setParameter(5, QVariant::fromValue<float>(4));

    f.flat(); // returns "[Key1=1, Key2=2, Arg1=3, Arg5=4]"

### bool operator ==(const char \*other) const

Compare [name](#name) to c-style string other.

    File f("picture.jpg");

    f == "picture.jpg";       // returns true
    f == "other_picture.jpg"; // returns false

### bool operator ==(const [File](#file) &other) const

Compare [name](#name) and [metadata](#m_metadata) to another file name and metadata for equality.

    File f1("picture1.jpg");
    File f2("picture1.jpg");

    f1 == f2; // returns true

    f1.set("Key1", QVariant::fromValue<float>(1));
    f2.set("Key2", QVariant::fromValue<float>(2));

    f1 == f2; // returns false (metadata doesn't match)

### bool operator !=(const [File](#file) &other) const

Compare [name](#name) and [metadata](#m_metadata) to another file name and metadata for inequality.

    File f1("picture1.jpg");
    File f2("picture1.jpg");

    f1 != f2; // returns false

    f1.set("Key1", QVariant::fromValue<float>(1));
    f2.set("Key2", QVariant::fromValue<float>(2));

    f1 != f2; // returns true (metadata doesn't match)

### bool operator <(const [File](#file) &other) const

Compare [name](#name) to a different file name.

### bool operator <=(const [File](#file) &other) const

Compare [name](#name) to a different file name.

### bool operator >(const [File](#file) &other) const

Compare [name](#name) to a different file name.

### bool operator >=(const [File](#file) &other) const

Compare [name](#name) to a different file name.

### bool isNull() const

Returns true if [name](#name) and [metadata](#m_metadata) are empty and false otherwise.

    File f;
    f.isNull(); // returns true

    f.set("Key1", QVariant::fromValue<float>(1));
    f.isNull(); // returns false

### bool isTerminal() const

Returns true if [name](#name) equals "Terminal".

### bool exists() const

Returns true if the file at [name](#name) exists on disk.

### [QString](http://doc.qt.io/qt-5/QString.html) fileName() const

Returns the file's base name and extension.

    File file("../path/to/pictures/picture.jpg");
    file.fileName(); // returns "picture.jpg"

### [QString](http://doc.qt.io/qt-5/QString.html) baseName() const

Returns the file's base name.

    File file("../path/to/pictures/picture.jpg");
    file.baseName(); // returns "picture"

### [QString](http://doc.qt.io/qt-5/QString.html) suffix() const

Returns the file's extension.

    File file("../path/to/pictures/picture.jpg");
    file.suffix(); // returns "jpg"

### [QString](http://doc.qt.io/qt-5/QString.html) path() const

Return's the path of the file, excluding the name.

    File file("../path/to/pictures/picture.jpg");
    file.suffix(); // returns "../path/to/pictures"

### [QString](http://doc.qt.io/qt-5/QString.html) resolved() const

Returns [name](#name). If name does not exist it prepends name with the path in Globals->path.

### bool contains(const [QString](http://doc.qt.io/qt-5/QString.html) &key) const

Returns True if the key is in the [metadata](#m_metadata) and False otherwise.

    File file;
    file.set("Key1", QVariant::fromValue<float>(1));

    file.contains("Key1"); // returns true
    file.contains("Key2"); // returns false

### bool contains(const [QStringList](http://doc.qt.io/qt-4.8/qstringlist.html) &keys) const

Returns True if all of the keys are in the [metadata](#m_metadata) and False otherwise.

    File file;
    file.set("Key1", QVariant::fromValue<float>(1));
    file.set("Key2", QVariant::fromValue<float>(2));

    file.contains(QStringList("Key1")); // returns true
    file.contains(QStringList() << "Key1" << "Key2") // returns true
    file.contains(QStringList() << "Key1" << "Key3"); // returns false

### [QVariant](http://doc.qt.io/qt-5/qvariant.html) value(const [QString](http://doc.qt.io/qt-5/QString.html) &key) const

Returns the value associated with key in the [metadata](#m_metadata).

    File file;
    file.set("Key1", QVariant::fromValue<float>(1));
    file.value("Key1"); // returns QVariant(float, 1)

### void set(const [QString](http://doc.qt.io/qt-5/QString.html) &key, const [QVariant](http://doc.qt.io/qt-5/qvariant.html) &value)

Insert or overwrite the [metadata](#m_metadata) key with the given value.

    File f;
    f.flat(); // returns ""

    f.set("Key1", QVariant::fromValue<float>(1));
    f.flat(); // returns "[Key1=1]"

### void set(const [QString](http://doc.qt.io/qt-5/QString.html) &key, const [QString](http://doc.qt.io/qt-5/QString.html) &value)

Insert or overwrite the [metadata](#m_metadata) key with the given value.

    File f;
    f.flat(); // returns ""

    f.set("Key1", QString("1"));
    f.flat(); // returns "[Key1=1]"

### void setList(const [QString](http://doc.qt.io/qt-5/QString.html) &key, const [QList](http://doc.qt.io/qt-5/qlist.html)&lt;T&gt; &value)

This function requires a type specification in place of T. Insert or overwrite the [metadata](#m_metadata) key with the value. The value will remain a list and should be queried with the function [getList](#qlistt-getlistconst-qstring-key-const).

    File file;

    QList<float> list = QList<float>() << 1 << 2 << 3;
    file.setList<float>("List", list);
    file.getList<float>("List"); // return [1., 2. 3.]

### void remove(const [QString](http://doc.qt.io/qt-5/QString.html) &key)

Remove the key value pair associated with the given key from the [metadata](#metadata)

    File f;
    f.set("Key1", QVariant::fromValue<float>(1));
    f.set("Key2", QVariant::fromValue<float>(2));

    f.flat(); // returns "[Key1=1, Key2=2]"

    f.remove("Key1");
    f.flat(); // returns "[Key2=2]"

### T get(const [QString](http://doc.qt.io/qt-5/QString.html) &key)

This function requires a type specification in place of T. Try and get the value associated with the given key in the [metadata](#m_metadata). If the key does not exist or cannot be converted to the given type an error is thrown.

    File f;
    f.set("Key1", QVariant::fromValue<float>(1));

    f.get<float>("Key1");  // returns 1
    f.get<float>("Key2");  // Error: Key2 is not in the metadata
    f.get<QRectF>("Key1"); // Error: A float can't be converted to a QRectF

### T get(const [QString](http://doc.qt.io/qt-5/QString.html) &key, const T &defaultValue)

This function requires a type specification in place of T. Try and get the value associated with the given key in the [metadata](#m_metadata). If the key does not exist or cannot be converted to the given type the defaultValue is returned.

    File f;
    f.set("Key1", QVariant::fromValue<float>(1));

    f.get<float>("Key1", 5);  // returns 1
    f.get<float>("Key2", 5);  // returns 5
    f.get<QRectF>("Key1", QRectF(0, 0, 10, 10)); // returns QRectF(0, 0, 10x10)

### bool getBool(const [QString](http://doc.qt.io/qt-5/QString.html) &key, bool defaultValue = false)

This is a specialization of [get](#t-getconst-qstring-key) for the boolean type. If the key is not in the [metadata](#m_metadata) the defaultValue is returned. If the key is in the metadata but the value cannot be converted to a bool **true** is returned. If the key is found and the value can be converted to a bool the value is returned.

    File f;
    f.set("Key1", QVariant::fromValue<bool>(true));
    f.set("Key2", QVariant::fromValue<float>(10));

    f.getBool("Key1");       // returns true
    f.getBool("Key2")        // returns true (key found)
    f.getBool("Key3");       // returns false (default value)
    f.getBool("Key3", true); // returns true (default value)

### [QList](http://doc.qt.io/qt-5/QList.html)&lt;T&gt; getList(const [QString](http://doc.qt.io/qt-5/QString.html) &key) const

This function requires a type specification in place of T. Similar to [get](#t-getconst-qstring-key) only this returns a list. If the key is not found or the value cannot be converted into a [QList](http://doc.qt.io/qt-5/QList.html)&lt;T&gt; an error is thrown.

    File file;

    QList<float> list = QList<float>() << 1 << 2 << 3;
    file.setList<float>("List", list);

    file.getList<float>("List");  // return [1., 2. 3.]
    file.getList<QRectF>("List"); // Error: float cannot be converted to QRectF
    file.getList<float>("Key");   // Error: key doesn't exist

### [QList](http://doc.qt.io/qt-5/QList.html)&lt;T&gt; getList(const [QString](http://doc.qt.io/qt-5/QString.html) &key, const [QList](http://doc.qt.io/qt-5/QList.html)&lt;T&gt; &defaultValue) const

This function requires a type specification in place of T. Similar to [get](#t-getconst-qstring-key-const-t-defaultvalue) only this returns a list. If the key is not found or the value cannot be converted into a [QList](http://doc.qt.io/qt-5/QList.html)&lt;T&gt; the supplied defaultValue is returned.

    File file;

    QList<float> list = QList<float>() << 1 << 2 << 3;
    file.setList<float>("List", list);

    file.getList<float>("List", QList<float>());                  // return [1., 2. 3.]
    file.getList<QRectF>("List", QList<QRectF>());                // return []
    file.getList<float>("Key", QList<float>() << 1 << 2 << 3);    // return [1., 2., 3.]

### [QList](http://doc.qt.io/qt-5/QList.html)&lt;[QPointF](http://doc.qt.io/qt-4.8/qpointf.html)&gt; namedPoints() const

Find all of the points that can be parsed from [metadata](#m_metadata) keys and return them. Only values that are convertable to [QPointF](http://doc.qt.io/qt-4.8/qpointf.html) are found. Values that can be converted to [QList](http://doc.qt.io/qt-5/QList.html)>&lt;[QPointF](http://doc.qt.io/qt-4.8/qpointf.html)&gt; are not included.

    File file;
    file.set("Key1", QVariant::fromValue<QPointF>(QPointF(1, 1)));
    file.set("Key2", QVariant::fromValue<QPointF>(QPointF(2, 2)));
    file.set("Points", QVariant::fromValue<QPointF>(QPointF(3, 3)))

    f.namedPoints(); // returns [QPointF(1, 1), QPointF(2, 2), QPointF(3, 3)]

    file.setPoints(QList<QPointF>() << QPointF(3, 3)); // changes metadata["Points"] to QList<QPointF>
    f.namedPoints(); // returns [QPointF(1, 1), QPointF(2, 2)]

### [QList](http://doc.qt.io/qt-5/QList.html)&lt;[QPointf](http://doc.qt.io/qt-4.8/qpointf.html)>&gt; points() const

Returns the list of points stored in [metadata](#m_metadata)["Points"]. A list is expected and a single point not in a list will not be returned. Convenience functions [appendPoint](#void-appendpointconst-qpointf-point), [appendPoints](#void-appendpointsconst-qlistqpointf-points), [clearPoints](#void-clearpoints) and [setPoints](#void-setpointsconst-qlistqpointf-points) have been provided to manipulate the internal points list.

    File file;
    file.set("Points", QVariant::fromValue<QPointF>(QPointF(1, 1)));
    file.points(); // returns [] (point is not in a list)

    file.setPoints(QList<QPointF>() << QPointF(2, 2));
    file.points(); // returns [QPointF(2, 2)]

### void appendPoint(const [QPointF](http://doc.qt.io/qt-4.8/qpointf.html) &point)

Add a point to the file's points list stored in [metadata](#m_metadata)["Points"]

    File file;
    file.points(); // returns []

    file.appendPoint(QPointF(1, 1));
    file.points(); // returns [QPointF(1, 1)]

### void appendPoints(const [QList](http://doc.qt.io/qt-5/QList.html)&lt;[QPointF](http://doc.qt.io/qt-4.8/qpointf.html)&gt; &points)

Add a list of points to the file's points list stored in [metadata](#m_metadata)["Points"]

    File file;
    file.points(); // returns []

    file.appendPoints(QList<QPointF>() << QPointF(1, 1) << QPointF(2, 2));
    file.points(); // returns [QPointF(1, 1), QPointF(2, 2)]

### void clearPoints()

Clear the list of points stored in [metadata](#m_metadata)["Points"].

    File file;
    file.appendPoints(QList<QPointF>() << QPointF(1, 1) << QPointF(2, 2));
    file.points(); // returns [QPointF(1, 1), QPointF(2, 2)]

    file.clearPoints();
    file.points(); // returns []

### void setPoints(const [QList](http://doc.qt.io/qt-5/QList.html)&lt;[QPointF](http://doc.qt.io/qt-4.8/qpointf.html)&gt; &points)

Clears the points stored in [metadata](#m_metadata) and replaces them with points.

    File file;
    file.appendPoints(QList<QPointF>() << QPointF(1, 1) << QPointF(2, 2));
    file.points(); // returns [QPointF(1, 1), QPointF(2, 2)]

    file.setPoints(QList<QPointF>() << QPointF(3, 3) << QPointF(4, 4));
    file.points(); // returns [QPointF(3, 3), QPointF(4, 4)]

### [QList](http://doc.qt.io/qt-5/QList.html)&lt;[QRectF](http://doc.qt.io/qt-4.8/qrectf.html)&gt; namedRects() const

Find all of the rects that can be parsed from [metadata](#m_metadata) keys and return them. Only values that are convertable to [QRectF](http://doc.qt.io/qt-4.8/qrectf.html) are found. Values that can be converted to [QList](http://doc.qt.io/qt-5/QList.html)&lt;[QRectF](http://doc.qt.io/qt-4.8/qrectf.html)&gt; are not included.

    File file;
    file.set("Key1", QVariant::fromValue<QRectF>(QRectF(1, 1, 5, 5)));
    file.set("Key2", QVariant::fromValue<QRectF>(QRectF(2, 2, 5, 5)));
    file.set("Rects", QVariant::fromValue<QRectF>(QRectF(3, 3, 5, 5)));

    f.namedRects(); // returns [QRectF(1, 1, 5x5), QRectF(2, 2, 5x5), QRectF(3, 3, 5x5)]

    file.setRects(QList<QRectF>() << QRectF(3, 3, 5x5)); // changes metadata["Rects"] to QList<QRectF>
    f.namedRects(); // returns [QRectF(1, 1, 5x5), QRectF(2, 2, 5x5)]

### [QList](http://doc.qt.io/qt-5/QList.html)&lt;[QRectF](http://doc.qt.io/qt-4.8/qrectf.html)&gt; rects() const

Returns the list of points stored in [metadata](#m_metadata)["Rects"]. A list is expected and a single rect not in a list will not be returned. Convenience functions [appendRect](#void-appendrectconst-qrectf-rect), [appendRects](#void-appendrectsconst-qlistqrectf-rects), [clearRects](#void-clearrects) and [setRects](#void-setrectsconst-qlistqrectf-rects) have been provided to manipulate the internal points list.

    File file;
    file.set("Rects", QVariant::fromValue<QRectF>(QRectF(1, 1, 5, 5)));
    file.rects(); // returns [] (rect is not in a list)

    file.setRects(QList<QRectF>() << QRectF(2, 2, 5, 5));
    file.rects(); // returns [QRectF(2, 2, 5x5)]

### void appendRect(const [QRectF](http://doc.qt.io/qt-4.8/qrectf.html) &rect)

Add a rect to the file's rects list stored in [metadata](#m_metadata)["Rects"].

    File file;
    file.rects(); // returns []

    file.appendRect(QRectF(1, 1, 5, 5));
    file.rects(); // returns [QRectF(1, 1, 5x5)]

### void appendRect(const [Rect](http://docs.opencv.org/modules/core/doc/basic_structures.html#rect) &rect)

Add a OpenCV style rect to the file's rects list stored in [metadata](#m_metadata)["Rects"]. The rect is automatically converted to a QRectF.

    File file;
    file.rects(); // returns []

    file.appendRect(cv::Rect(1, 1, 5, 5)); // automatically converted to QRectF
    file.rects(); // returns [QRectF(1, 1, 5x5)]

### void appendRects(const [QList](http://doc.qt.io/qt-5/QList.html)&lt;[QRectF](http://doc.qt.io/qt-4.8/qrectf.html)&gt; &rects)

Add a list of rects to the file's rects list stored in [metadata](#m_metadata)["Rects"]

    File file;
    file.rects(); // returns []

    file.appendRects(QList<QRectF>() << QRectF(1, 1, 5, 5) << QRectF(2, 2, 5, 5));
    file.rects(); // returns [QRectF(1, 1, 5x5), QRectF(2, 2, 5x5)]

### void appendRects(const [QList](http://doc.qt.io/qt-5/QList.html)&lt;[Rect](http://docs.opencv.org/modules/core/doc/basic_structures.html#rect)&gt; &rects)

Add a list of OpenCV style rects to the file's rects list stored in [metadata](#m_metadata)["Rects"]. Each rect is automatically converted to a QRectF.

    File file;
    file.rects(); // returns []

    file.appendRects(QList<cv::Rect>() << cv::Rect(1, 1, 5, 5) << cv::Rect(2, 2, 5, 5));
    file.rects(); // returns [QRectF(1, 1, 5x5), QRectF(2, 2, 5x5)]


### void clearRects()

Clear the list of rects stored in [metadata](#m_metadata)["Rects"].

    File file;
    file.appendRects(QList<QRectF>() << QRectF(1, 1, 5, 5) << QRectF(2, 2, 5, 5));
    file.rects(); // returns [QRectF(1, 1, 5x5), QRectF(2, 2, 5x5)]

    file.clearRects();
    file.rects(); // returns []

### void setRects(const [QList](http://doc.qt.io/qt-5/QList.html)&lt;[QRectF](http://doc.qt.io/qt-4.8/qrectf.html)&gt; &rects)

Clears the rects stored in [metadata](#m_metadata)["Rects"] and replaces them with the given rects.

    File file;
    file.appendRects(QList<QRectF>() << QRectF(1, 1, 5, 5) << QRectF(2, 2, 5, 5));
    file.rects(); // returns [QRectF(1, 1, 5x5), QRectF(2, 2, 5x5)]

    file.setRects(QList<QRectF>() << QRectF(3, 3, 5, 5) << QRectF(4, 4, 5, 5));
    file.rects(); // returns [QRectF(3, 3, 5x5), QRectF(4, 4, 5x5)]

### void setRects(const [QList](http://doc.qt.io/qt-5/QList.html)&lt;[Rect](http://docs.opencv.org/modules/core/doc/basic_structures.html#rect)&gt; &rects)

Clears the rects stored in [metadata](#m_metadata)["Rects"] and replaces them with the given OpenCV style rects.

    File file;
    file.appendRects(QList<cv::Rect>() << cv::Rect(1, 1, 5, 5) << cv::Rect(2, 2, 5, 5));
    file.rects(); // returns [QRectF(1, 1, 5x5), QRectF(2, 2, 5x5)]

    file.setRects(QList<cv::Rect>() << cv::Rect(3, 3, 5, 5) << cv::Rect(4, 4, 5, 5));
    file.rects(); // returns [QRectF(3, 3, 5x5), QRectF(4, 4, 5x5)]

---

# FileList

A convenience class for dealing with lists of files.

## Constructors

### FileList()

Default constructor. Doesn't do anything

### FileList(int n)

Initialize the [FileList](#filelist) with n empty files

### FileList(const [QStringList](http://doc.qt.io/qt-4.8/qstringlist.html) &files)

Initialize the [FileList](#filelist) from a list of strings. Each string should have the format "filename[key1=value1, key2=value2, ... keyN=valueN]"

### FileList(const [QList](http://doc.qt.io/qt-4.8/qlist.html)&lt;[File](#file)&gt; &files)

Initialize the [FileList](#filelist) from a list of [files](#file).

## Static Functions

### static FileList fromGallery(const [File](#file) &gallery, bool cache = false)

Creates a [FileList](#filelist) from a [Gallery](#gallery). Galleries store one or more [Templates](#template) on disk. Common formats include csv, xml, and gal, which is a unique OpenBR format. Read more about this in the [Gallery](#gallery) section. This function creates a [FileList](#filelist) by parsing the stored gallery based on its format. Cache determines whether the gallery should be stored for faster reading later.

    File gallery("gallery.csv");

    FileList fList = FileList::fromGallery(gallery);
    fList.flat(); // returns all the files that have been loaded from disk. It could
                  // be 1 or 100 depending on what was stored.

## Functions

### [QStringList](http://doc.qt.io/qt-4.8/qstringlist.html) flat() const

Calls [flat](#qstring-flat-const) on every [File](#file) in the list and returns the resulting strings as a [QStringList](http://doc.qt.io/qt-4.8/qstringlist.html).

    File f1("picture1.jpg"), f2("picture2.jpg");
    f1.set("Key", QString("Value"));

    FileList fList(QList<File>() << f1 << f2);
    fList.flat(); // returns ["picture1.jpg[Key=Value]", "picture2.jpg"]

### [QStringList](http://doc.qt.io/qt-4.8/qstringlist.html) names() const

Stores the name of every [file](#file) in the list and returns the resulting strings as a [QStringList](http://doc.qt.io/qt-4.8/qstringlist.html).

    File f1("picture1.jpg"), f2("picture2.jpg");
    f1.set("Key", QString("Value"));

    FileList fList(QList<File>() << f1 << f2);
    fList.names(); // returns ["picture1.jpg", "picture2.jpg"]

### void sort(const [QString](http://doc.qt.io/qt-4.8/qstring.html) &key)

Sorts the [FileList](#filelist) based on the value associated with the given key in each file.

    File f1("1"), f2("2"), f3("3");
    f1.set("Key", QVariant::fromValue<float>(3));
    f2.set("Key", QVariant::fromValue<float>(1));
    f3.set("Key", QVariant::fromValue<float>(2));

    FileList fList(QList<File>() << f1 << f2 << f3);
    fList.names(); // returns ["1", "2", "3"]

    fList.sort("Key");
    fList.names(); // returns ["2", "3", "1"]

---

# Template

---

# TemplateList

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
