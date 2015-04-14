<!-- FILE -->

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
The left hand side of the string not parsed in a manner described above is assigned to [name](#file-members-name).

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
name            | QString        | Contents of [name](#file-members-name)
separator       | QString        | Separate [name](#file-members-name) into multiple files
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

## Members {: #file-members }

Member | Type | Description
--- | --- | ---
<a class="table-anchor" id="file-members-name"></a>name | [QString][QString] | Path to a file on disk
<a class="table-anchor" id=file-members-fte></a>fte | bool | Failed to enroll. If true this file failed to be processed somewhere in the template enrollment algorithm
<a class="table-anchor" id=file-members-m_metadata></a>m_metadata | [QVariantMap][QVariantMap] | Map for storing metadata. It is a [QString][QString], [QVariant][QVariant] key value pairing.

---

## Constructors {: #file-constructors }

Constructor | Description
--- | ---
File() | Default constructor. Sets [name](#file-members-fte) to false.
File(const [QString][QString] &file) | Initializes the file by calling the private function init.
File(const [QString][QString] &file, const [QVariant][QVariant] &label) | Initializes the file by calling the private function init. Append label to the [metadata](#file-members-m_metadata) using the key "Label".
File(const char \*file) | Initializes the file with a c-style string.
File(const [QVariantMap][QVariantMap] &metadata) | Sets [name](#file-members-fte) to false and sets the [file metadata](#file-members) to metadata.

---

## Static Functions {: #file-static-functions }


### [QVariant][QVariant] parse(const [QString][QString] &value) {: #function-static-parse }

Try to convert a given value to a [QPointF][QPointF], [QRectF][QRectF], int or float.

* **function definition:**

        static QVariant parse(const QString &value);

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    value | const [QString][QString] & | String value to be converted.

* **output:** ([QVariant][QVariant]) The converted file if a conversion was possible. Otherwise the unconverted string is returned.
* **example:**

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


### [QList][QList]&lt;[QVariant][QVariant]&gt; values(const [QList][QList]&lt;U&gt; &fileList, const [QString][QString] &key) {: #file-static-values }

Gather a list of [QVariant][QVariant] values associated with a metadata key from a provided list of files.

* **function definition:**

        template<class U> static [QList<QVariant> values(const QList<U> &fileList, const QString &key)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    fileList | const [QList][QList]&lt;U&gt; & | A list of files to parse for values. A type is required for <tt>U</tt>. Valid options are: <ul> <li>[File](#file)</li> <li>[QString][QString]</li> </ul>
    key | const [QString][QString] & | A metadata key used to lookup the values.

* **output:** ([QList][QList]&lt;[QVariant][QVariant]&gt;) A list of [QVariant][QVariant] values associated with the given key in each of the provided files.
* **example:**

        File f1, f2;
        f1.set("Key1", QVariant::fromValue<float>(1));
        f1.set("Key2", QVariant::fromValue<float>(2));
        f2.set("Key1", QVariant::fromValue<float>(3));

        File::values<File>(QList<File>() << f1 << f2, "Key1"); // returns [QVariant(float, 1),
                                                               //          QVariant(float, 3)]


### [QList][QList]&lt;T&gt; get(const [QList][QList]&lt;U&gt; &fileList, const [QString][QString] &key) {: #file-static-get-1 }

Gather a list of <tt>T</tt> values associated with a metadata key from a provided list of files. <tt>T</tt> is a user provided type. If the key does not exist in the metadata of *any* file an error is thrown.

* **function definition:**

        template<class T, class U> static QList<T> get(const QList<U> &fileList, const QString &key)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    fileList | const [QList][QList]&lt;U&gt; & | A list of files to parse for values. A type is required for U. Valid options are: <ul> <li>[File](#file)</li> <li>[QString][QString]</li> </ul>
    key | const [QString][QString] & | A metadata key used to lookup the values.

* **output:** ([QList][QList]&lt;T&gt;) A list of the values of type <tt>T</tt> associated with the given key. A type is required for <tt>T</tt>.
* **example:**

        File f1, f2;
        f1.set("Key1", QVariant::fromValue<float>(1));
        f1.set("Key2", QVariant::fromValue<float>(2));
        f2.set("Key1", QVariant::fromValue<float>(3));

        File::get<float, File>(QList<File>() << f1 << f2, "Key1");  // returns [1., 3.]
        File::get<float, File>(QList<File>() << f1 << f2, "Key2");  // Error: Key doesn't exist in f2
        File::get<QRectF, File>(QList<File>() << f1 << f2, "Key1"); // Error: float is not convertable to QRectF


### [QList][QList]&lt;T&gt; get(const [QList][QList]&lt;U&gt; &fileList, const [QString][QString] &key, const T &defaultValue) {: #file-static-get-2 }

Gather a list of <tt>T</tt> values associated with a metadata key from a provided list of files. <tt>T</tt> is a user provided type. If the key does not exist in the metadata of *any* file the provided **defaultValue** is used.

* **function definition:**

        template<class T, class U> static QList<T> get(const QList<U> &fileList, const QString &key, const T &defaultValue)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    fileList | const [QList][QList]&lt;U&gt; & | A list of files to parse for values. A type is required for U. Valid options are: <ul> <li>[File](#file)</li> <li>[QString][QString]</li> </ul>
    key | const [QString][QString] & | A metadata key used to lookup the values.
    defaultValue | const T & | The default value if the key is not in a file's metadata. A type is required for T. All types are valid.

* **output:** ([QList][QList]&lt;T&gt;) A list of the values of type <tt>T</tt> associated with the given key. A type is required for <tt>T</tt>.
* **example:**

    File f1, f2;
    f1.set("Key1", QVariant::fromValue<float>(1));
    f1.set("Key2", QVariant::fromValue<float>(2));
    f2.set("Key1", QVariant::fromValue<float>(3));

    File::get<float, File>(QList<File>() << f1 << f2, "Key1");                       // returns [1., 3.]
    File::get<float, File>(QList<File>() << f1 << f2, "Key2", QList<float>() << 1);  // returns [1.]
    File::get<QRectF, File>(QList<File>() << f1 << f2, "Key1, QList<QRectF>()");     // returns []


### [QDebug][QDebug] operator <<([QDebug][QDebug] dbg, const [File](#file) &file) {: #file-static-dbg-operator-ltlt }

Calls [flat](#file-function-flat) on the given file and then streams that file to stderr.

* **function definition:**

        QDebug operator <<(QDebug dbg, const File &file)

* **parameter:**

    Parameter | Type | Description
    --- | --- | ---
    dbg | [QDebug][QDebug] | The debug stream
    file | const [File](#file) & | File to stream

* **output:** ([QDebug][QDebug] &) returns a reference to the updated debug stream
* **example:**

        File file("../path/to/pictures/picture.jpg");
        file.set("Key", QString("Value"));

        qDebug() << file; // "../path/to/pictures/picture.jpg[Key=Value]" streams to stderr


### [QDataStream][QDataStream] &operator <<([QDataStream][QDatastream] &stream, const [File](#file) &file) {: #file-static-stream-operator-ltlt }

Serialize a file to a data stream.

* **function definition:**

        QDataStream &operator <<(QDataStream &stream, const File &file)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    stream | [QDataStream][QDataStream] | The data stream
    file | const [File](#file) & | File to stream

* **output:** ([QDataStream][QDataStream] &) returns a reference to the updated data stream
* **example:**

        void store(QDataStream &stream)
        {
            File file("../path/to/pictures/picture.jpg");
            file.set("Key", QString("Value"));

            stream << file; // "../path/to/pictures/picture.jpg[Key=Value]" serialized to the stream
        }


### [QDataStream][QDataStream] &operator >>([QDataStream][QDataStream] &stream, const [File](#file) &file) {: #file-static-stream-operator-gtgt }

Deserialize a file from a data stream.

* **function definition:**

        QDataStream &operator >>(QDataStream &stream, const File &file)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    stream | [QDataStream][QDataStream] | The data stream
    file | const [File](#file) & | File to stream

* **output:** ([QDataStream][QDataStream] &) returns a reference to the updated data stream
* **example:**

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

## Functions {: #file-functions }


### operator [QString][QString]() const {: #file-function-operator-qstring }

Convenience function that allows [Files](#file) to be used as [QStrings][QString]

* **function definition:**

        inline operator QString() const

* **parameters:** NONE
* **output:** ([QString][QString]) returns [name](#file-members-name).

### [QString][QString] flat() const {: #file-function-flat }

Function to output files in string formats.

* **function definition:**

        QString flat() const

* **parameters:** NONE
* **output:** ([QString][QString]) returns the [file name](#file-members) and [metadata](#file-members-m_metadata) as a formated string. The format is *filename*[*key1=value1,key2=value2,...keyN=valueN*].
* **example:**

        File file("picture.jpg");
        file.set("Key1", QVariant::fromValue<float>(1));
        file.set("Key2", QVariant::fromValue<float>(2));

        file.flat(); // returns "picture.jpg[Key1=1,Key2=2]"


### [QString][QString] hash() const {: #file-function-hash }

Function to output a hash of the file.

* **function definition:**

        QString hash() const

* **parameters:** NONE
* **output:** ([QString][QString]) Returns a hash of the file.
* **example:**

        File file("../path/to/pictures/picture.jpg");
        file.set("Key1", QVariant::fromValue<float>(1));
        file.set("Key2", QVariant::fromValue<float>(2));

        file.hash(); // returns "kElVwY"


### [QStringList][QStringList] localKeys() const {: #file-function-localkeys }

Function to get the private [metadata](#file-members-m_metadata) keys.

* **function definition:**

        inline QStringList localKeys() const

* **parameters:** NONE
* **output:** ([QStringList][QStringList]) Returns a list of the local [metadata](#file-members-m_metadata) keys. They are called local because they do not include the keys in the [global metadata](#context).
* **example:**

    File file("../path/to/pictures/picture.jpg");
    file.set("Key1", QVariant::fromValue<float>(1));
    file.set("Key2", QVariant::fromValue<float>(2));

    file.localKeys(); // returns [Key1, Key2]


### [QVariantMap][QVariantMap] localMetadata() const {: #file-function-localmetadata }

Function to get the private [metadata](#file-members-m_metadata).

* **function definition:**

        inline QVariantMap localMetadata() const

* **parameters:** NONE
* **output:** ([QVariantMap][QVariantMap]) Returns the local [metadata](#file-members-m_metadata).
* **example:**

        File file("../path/to/pictures/picture.jpg");
        file.set("Key1", QVariant::fromValue<float>(1));
        file.set("Key2", QVariant::fromValue<float>(2));

        file.localMetadata(); // return QMap(("Key1", QVariant(float, 1)) ("Key2", QVariant(float, 2)))


### void append(const [QVariantMap][QVariantMap] &localMetadata) {: #file-function-append-1 }

Add new metadata fields to [metadata](#file-members-m_metadata).

* **function definition:**

        void append(const QVariantMap &localMetadata)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    localMetadata | const [QVariantMap][QVariantMap] & | metadata to append to the local [metadata](#file-members-m_metadata)

* **output:** (void)
* **example:**

        File f();
        f.set("Key1", QVariant::fromValue<float>(1));

        QVariantMap map;
        map.insert("Key2", QVariant::fromValue<float>(2));
        map.insert("Key3", QVariant::fromValue<float>(3));

        f.append(map);
        f.flat(); // returns "[Key1=1, Key2=2, Key3=3]"


### void append(const [File](#file) &other) {: #file-function-append-2 }

Append another file using the **;** separator. The [File](#file) [names](#file-members-name) are combined with the separator in between them. The [metadata](#file-members-m_metadata) fields are combined. An additional field describing the separator is appended to the [metadata](#file-members-m_metadata).

* **function definition:**

        void append(const File &other)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    other | const [File](#file) & | File to append

* **output:** (void)
* **example:**

        File f1("../path/to/pictures/picture1.jpg");
        f1.set("Key1", QVariant::fromValue<float>(1));

        File f2("../path/to/pictures/picture2.jpg");
        f2.set("Key2", QVariant::fromValue<float>(2));
        f2.set("Key3", QVariant::fromValue<float>(3));

        f1.append(f2);
        f1.name; // return "../path/to/pictures/picture1.jpg;../path/to/pictures/picture2.jpg"
        f1.localKeys(); // returns "[Key1, Key2, Key3, separator]"


### [File](#file) &operator+=(const [QMap][QMap]&lt;[QString][QString], [QVariant][QVariant]&gt; &other) {: #file-function-operator-pe-1 }

Shortcut operator to call [append](#file-function-append-1).

* **function definition:**

        inline File &operator+=(const QMap<QString, QVariant> &other)

* **parameters:**

    Parameter | Type | Description
    other | const [QMap][QMap]&lt;[QString][QString], [QVariant][QVariant]&gt; & | Metadata map to append to the local [metadata](#file-members-m_metadata)

* **output:** ([File](#file) &) Returns a reference to this file after the append occurs.
* **example:**

        File f();
        f.set("Key1", QVariant::fromValue<float>(1));

        QMap<QString, QVariant> map;
        map.insert("Key2", QVariant::fromValue<float>(2));
        map.insert("Key3", QVariant::fromValue<float>(3));

        f += map;
        f.flat(); // returns "[Key1=1, Key2=2, Key3=3]"


### [File](#file) &operator+=(const [File](#file) &other) {: #file-function-operator-pe-2 }

Shortcut operator to call [append](#file-function-append-2).

* **function definition:**

        inline File &operator+=(const File &other)

* **parameters:**

    Parameter | Type | Description
    other | const [File](#file) & | File to append

* **output:** ([File](#file) &) Returns a reference to this file after the append occurs.
* **example:**

        File f1("../path/to/pictures/picture1.jpg");
        f1.set("Key1", QVariant::fromValue<float>(1));

        File f2("../path/to/pictures/picture2.jpg");
        f2.set("Key2", QVariant::fromValue<float>(2));
        f2.set("Key3", QVariant::fromValue<float>(3));

        f1 += f2;
        f1.name; // return "../path/to/pictures/picture1.jpg;../path/to/pictures/picture2.jpg"
        f1.localKeys(); // returns "[Key1, Key2, Key3, separator]"


### [QList][QList]&lt;[File](#file)&gt; split() const {: #file-function-split-1 }

This function splits the [File](#file) into multiple files and returns them as a list. This is done by parsing the file [name](#file-members-name) and splitting on the separator located at [metadata](#file-members-m_metadata)["separator"]. If "separator" is not a [metadata](#file-members-m_metadata) key, the returned list has the original file as the only entry. Each new file has the same [metadata](#file-members-m_metadata) as the original, pre-split, file.

* **function definition:**

        QList<File> split() const

* **parameters:** None
* **output:** ([QList][QList]&lt;[File](#file)&gt;) List of split files
* **example:**

        File f1("../path/to/pictures/picture1.jpg");
        f1.set("Key1", QVariant::fromValue<float>(1));

        f1.split(); // returns [../path/to/pictures/picture1.jpg[Key1=1]]

        File f2("../path/to/pictures/picture2.jpg");
        f2.set("Key2", QVariant::fromValue<float>(2));
        f2.set("Key3", QVariant::fromValue<float>(3));

        f1.append(f2);
        f1.split(); // returns [../path/to/pictures/picture1.jpg[Key1=1, Key2=2, Key3=3, separator=;],
                    //          ../path/to/pictures/picture2.jpg[Key1=1, Key2=2, Key3=3, separator=;]]


### [QList][QList]&lt;[File](#file)&gt; split(const [QString][QString] &separator) const {: #file-function-split-2 }

This function splits the file into multiple files and returns them as a list. This is done by parsing the file [name](#file-members-name) and splitting on the given separator. Each new file has the same [metadata](#file-members-m_metadata) as the original, pre-split, file.

* **function definition:**

        QList<File> split(const QString &separator) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    separator | const [QString][QString] & | Separator to split the file name on

* **output:** ([QList][QList]&lt;[File](#file)&gt;) List of split files
* **example:**

        File f("../path/to/pictures/picture1.jpg,../path/to/pictures/picture2.jpg");
        f.set("Key1", QVariant::fromValue<float>(1));
        f.set("Key2", QVariant::fromValue<float>(2));

        f.split(","); // returns [../path/to/pictures/picture1.jpg[Key1=1, Key2=2],
                                  ../path/to/pictures/picture2.jpg[Key1=1, Key2=2]]


### void setParameter(int index, const [QVariant][QVariant] &value) {: #file-function-setparameter }

Insert a keyless value into the [metadata](#file-members-m_metadata). Generic key of "ArgN" is used, where N is given as a parameter.

* **function definition:**

        inline void setParameter(int index, const QVariant &value)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    index | int | Number to append to generic key
    value | const [QVariant][QVariant] & | Value to add to the metadata

* **output:** (void)
* **see:** [containsParameter](#file-function-containsparameter), [getParameter](#file-function-getparameter)
* **example:**

        File f;
        f.set("Key1", QVariant::fromValue<float>(1));
        f.set("Key2", QVariant::fromValue<float>(2));

        f.setParameter(1, QVariant::fromValue<float>(3));
        f.setParameter(5, QVariant::fromValue<float>(4));

        f.flat(); // returns "[Key1=1, Key2=2, Arg1=3, Arg5=4]"


### bool containsParameter(int index) const {: #file-function-containsparameter }

Check if the local [metadata](#file-members-m_metadata) contains a keyless value.

* **function definition:**

        inline bool containsParameter(int index) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    index | int | Index of the keyless value to check for

* **output:** (bool) Returns true if the local [metadata](#file-members-m_metadata) contains the keyless value, otherwise reutrns false.
* **see:** [setParameter](#file-function-setparameter), [getParameter](#file-function-getparameter)
* **example:**

        File f;
        f.setParameter(1, QVariant::fromValue<float>(1));
        f.setParameter(2, QVariant::fromValue<float>(2));

        f.containsParameter(1); // returns true
        f.containsParameter(2); // returns true
        f.containsParameter(3); // returns false


### [QVariant][QVariant] getParameter(int index) const {: #file-function-getparameter }

Get a keyless value from the local [metadata](#file-members-m_metadata). If the value does not exist an error is thrown.

* **function definition:**

        inline QVariant getParameter(int index) const

* **parameter:**

    Parameter | Type | Description
    --- | --- | ---
    index | int | Index of the keyless value to look up. If the index does not exist an error is thrown.

* **output:** ([QVariant][QVariant]) Returns the keyless value associated with the given index
* **see:** [setParameter](#file-function-setparameter), [containsParameter](#file-function-containsparameter)
* **example:**

        File f;
        f.setParameter(1, QVariant::fromValue<float>(1));
        f.setParameter(2, QVariant::fromValue<float>(2));

        f.getParameter(1); // returns 1
        f.getParameter(2); // returns 2
        f.getParameter(3); // error: index does not exist


### bool operator==(const char \*other) const {: #file-function-operator-ee-1 }

Compare [name](#file-members-name) against a c-style string.

* **function definition:**

        inline bool operator==(const char *other) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    other | const char \* | C-style string to compare against

* **output:** (bool) Returns true if the strings are equal, false otherwise.
* **example:**

        File f("picture.jpg");

        f == "picture.jpg";       // returns true
        f == "other_picture.jpg"; // returns false


### bool operator==(const [File](#file) &other) const {: #file-function-operator-ee-2 }

Compare [name](#file-members-name) and [metadata](#file-members-m_metadata) against another file name and metadata.

* **function definition:**

        inline bool operator==(const File &other) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    other | const [File](#file) & | File to compare against

* **output:** (bool) Returns true if the names and metadata are equal, false otherwise.
* **example:**

        File f1("picture1.jpg");
        File f2("picture1.jpg");

        f1 == f2; // returns true

        f1.set("Key1", QVariant::fromValue<float>(1));
        f2.set("Key2", QVariant::fromValue<float>(2));

        f1 == f2; // returns false (metadata doesn't match)


### bool operator!=(const [File](#file) &other) const {: #file-function-operator-ne }

Compare [name](#file-members-name) and [metadata](#file-members-m_metadata) against another file name and metadata.

* **function definition:**

        inline bool operator!=(const File &other) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    other | const [File](#file) & | File to compare against

* **output:** (bool) Returns true if the names and metadata are not equal, false otherwise.
* **example:**

        File f1("picture1.jpg");
        File f2("picture1.jpg");

        f1 != f2; // returns false

        f1.set("Key1", QVariant::fromValue<float>(1));
        f2.set("Key2", QVariant::fromValue<float>(2));

        f1 != f2; // returns true (metadata doesn't match)


### bool operator<(const [File](#file) &other) const {: #file-function-operator-lt }

Compare [name](#file-members-name) against another file name.

* **function definition:**

        inline bool operator<(const File &other) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    other | const [File](#file) & | File to compare against

* **output:** (bool) Returns true if [name](#file-members-name) < others.name


### bool operator<=(const [File](#file) &other) const {: #file-function-operator-lte }

Compare [name](#file-members-name) against another file name.

* **function definition:**

        inline bool operator<=(const File &other) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    other | const [File](#file) & | File to compare against

* **output:** (bool) Returns true if [name](#file-members-name) <= others.name


### bool operator>(const [File](#file) &other) const {: #file-function-operator-gt }

Compare [name](#file-members-name) against another file name.

* **function definition:**

        inline bool operator>(const File &other) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    other | const [File](#file) & | File to compare against

* **output:** (bool) Returns true if [name](#file-members-name) > others.name


### bool operator>=(const [File](#file) &other) const {: #file-function-operator-gte }

Compare [name](#file-members-name) against another file name.

* **function definition:**

        inline bool operator>=(const File &other) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    other | const [File](#file) & | File to compare against

* **output:** (bool) Returns true if [name](#file-members-name) >= others.name


### bool isNull() const {: #file-function-isnull }

Check if the file is null.

* **function definition:**

        inline bool isNull() const

* **parameters:** NONE
* **output:** (bool) Returns true if [name](#file-members-name) and [metadata](#file-members-m_metadata) are empty, false otherwise.
* **example:**

        File f;
        f.isNull(); // returns true

        f.set("Key1", QVariant::fromValue<float>(1));
        f.isNull(); // returns false


### bool isTerminal() const {: #file-function-isterminal }

Checks if the value of [name](#file-members-name) == "terminal".

* **function definition:**

        inline bool isTerminal() const

* **parameters:** NONE
* **output:** (bool) Returns true if [name](#file-members-name) == "terminal", false otherwise.
* **example:**

        File f1("terminal"), f2("not_terminal");

        f1.isTerminal(); // returns true
        f2.isTerminal(); // returns false


### bool exists() const {: #file-function-exists }

Check if the file exists on disk.

* **function definition:**

        inline bool exists() const

* **parameters:** NONE
* **output:** Returns true if [name](#file-members-name) exists on disk, false otherwise.
* **example:**

    File f1("/path/to/file/that/exists"), f2("/path/to/non/existant/file");

    f1.exists(); // returns true
    f2.exists(); // returns false


### [QString][QString] fileName() const {: #file-function-filename }

Get the file's base name and extension.

* **function definition:**

        inline QString fileName() const

* **parameters:** NONE
* **output:** ([QString][QString]) Returns the base name + extension of [name](#file-members-name)
* **example:**

        File file("../path/to/pictures/picture.jpg");
        file.fileName(); // returns "picture.jpg"


### [QString][QString] baseName() const {: #file-function-basename }

Get the file's base name.

* **function definition:**

        inline QString baseName() const

* **parameters:** NONE
* **output:** ([QString][QString]) Returns the base name of [name](#file-members-name)
* **example:**

        File file("../path/to/pictures/picture.jpg");
        file.baseName(); // returns "picture"


### [QString][QString] suffix() const {: #file-function-suffix }

Get the file's extension.

* **function definition:**

        inline QString suffix() const

* **parameters:** NONE
* **output:** ([QString][QString]) Returns the extension of [name](#file-members-name)
* **example:**

        File file("../path/to/pictures/picture.jpg");
        file.suffix(); // returns "jpg"


### [QString][QString] path() const {: #file-function-path }

Get the path of the file without the name.

* **function definition:**

        inline QString path() const

* **parameters:** NONE
* **output:** ([QString][QString]) Returns the path of [name](#file-members-name).
* **example:**

        File file("../path/to/pictures/picture.jpg");
        file.suffix(); // returns "../path/to/pictures"


### [QString][QString] resolved() const {: #file-function-resolved }

Get the full path for the file. This is done in three steps:

1. If [name](#file-members-name) exists, return [name](#file-members-name).
2. Prepend each path stored in [Globals->path](#context-members-path) to [name](#file-members-name). If the combined name exists then it is returned.
3. Prepend each path stored in [Globals->path](#context-members-path) to [fileName](#file-function-filename). If the combined name exists then it is returned.

If none of the attempted names exist, [name](#file-members-name) is returned unmodified.

* **function definition:**

        QString resolved() const

* **parameters:** NONE
* **output:** ([QString][QString]) Returns the resolved string if it can be created. Otherwise it returns [name](#file-members-name)


### bool contains(const [QString][QString] &key) const {: #file-function-contains-1 }

Check if a given key is in the local [metadata](#file-members-m_metadata).

* **function definition:**

        bool contains(const QString &key) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | Key to check the [metadata](#file-members-m_metadata) for

* **output:** (bool) Returns true if the given key is in the [metadata](#file-members-m_metadata), false otherwise.
* **example:**

        File file;
        file.set("Key1", QVariant::fromValue<float>(1));

        file.contains("Key1"); // returns true
        file.contains("Key2"); // returns false


### bool contains(const [QStringList][QStringList] &keys) const {: #file-function-contains-2 }

Check if a list of keys is in the local [metadata](#file-members-m_metadata).

* **function definition:**

        bool contains(const QStringList &keys) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    keys | const [QStringList][QStringList] & | Keys to check the [metadata](#file-members-m_metadata) for

* **output:** (bool) Returns true if *all* of the given keys are in the [metadata](#file-members-m_metadata), false otherwise.
* **example:**

        File file;
        file.set("Key1", QVariant::fromValue<float>(1));
        file.set("Key2", QVariant::fromValue<float>(2));

        file.contains(QStringList("Key1")); // returns true
        file.contains(QStringList() << "Key1" << "Key2") // returns true
        file.contains(QStringList() << "Key1" << "Key3"); // returns false


### [QVariant][QVariant] value(const [QString][QString] &key) const {: #file-function-value }

Get the value associated with a given key from the [metadata](#file-members-m_metadata). If the key is not found in the [local metadata](#file-members-m_metadata), the [global metadata](#context) is searched. In a special case, the key can be "name". This returns the file's [name](#file-members-name).

* **function description:**

        QVariant value(const QString &key) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | Key to look up the value in the [local metadata](#file-members-m_metadata) or [global metadata](#context). The key can also be "name".

* **output:** ([QVariant][QVariant]) Returns the key associated with the value from either the [local](#file-members-m_metadata) or [global](#context) metadata. If the key is "name", [name](#file-members-name) is returned.
* **example:**

        File file;
        file.set("Key1", QVariant::fromValue<float>(1));
        file.value("Key1"); // returns QVariant(float, 1)


### void set(const [QString][QString] &key, const [QVariant][QVariant] &value) {: #file-function-set-1 }

Insert a value into the [metadata](#file-members-m_metadata) using a provided key. If the key already exists the new value will override the old one.

* **function description:**

        inline void set(const QString &key, const QVariant &value)

* **parameters:**

    Parameters | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | Key to store the given value in the [metadata](#file-members-m_metadata)
    value | const [QVariant][QVariant] & | Value to be stored

* **output:** (void)
* **example:**

        File f;
        f.flat(); // returns ""

        f.set("Key1", QVariant::fromValue<float>(1));
        f.flat(); // returns "[Key1=1]"


### void set(const [QString][QString] &key, const [QString][QString] &value) {: #file-function-set-2 }

Insert a value into the [metadata](#file-members-m_metadata) using a provided key. If the key already exists the new value will override the old one.

* **function description:**

        void set(const QString &key, const QString &value)

* **parameters:**

    Parameters | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | Key to store the given value in the [metadata](#file-members-m_metadata)
    value | const [QString][QString] & | Value to be stored

* **output:** (void)
* **example:**

        File f;
        f.flat(); // returns ""

        f.set("Key1", QString("1"));
        f.flat(); // returns "[Key1=1]"


### void setList(const [QString][QString] &key, const [QList][QList]&lt;T&gt; &value) {: #file-function-setlist }

Insert a list into the [metadata](#file-members-m_metadata) using a provided key. If the key already exists the new value will override the old one. The value should be queried with [getList](#file-function-getlist-1) instead of [get](#file-function-get-1).

* **function description:**

        template <typename T> void setList(const QString &key, const QList<T> &value)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | Key to store the given value in the [metadata](#file-members-m_metadata)
    value | const [QList][QList]&lt;T&gt; | List to be stored

* **output:** (void)
* **see:** [getList](#file-function-getlist-1), [get](#file-function-get-1)
* **example:**

        File file;

        QList<float> list = QList<float>() << 1 << 2 << 3;
        file.setList<float>("List", list);
        file.getList<float>("List"); // return [1., 2. 3.]


### void remove(const [QString][QString] &key) {: #file-function-remove }

Remove a key-value pair from the [metadata](#file-members-m_metadata)

* **function description:**

        inline void remove(const QString &key)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | Key to be removed from [metadata](#file-members-m_metadata) along with its associated value.

* **output:** (void)
* **example:**

        File f;
        f.set("Key1", QVariant::fromValue<float>(1));
        f.set("Key2", QVariant::fromValue<float>(2));

        f.flat(); // returns "[Key1=1, Key2=2]"

        f.remove("Key1");
        f.flat(); // returns "[Key2=2]"


### T get(const [QString][QString] &key) const {: #file-function-get-1 }

Get a value from the [metadata](#file-members-m_metadata) using a provided key. If the key does not exist or the value cannot be converted to a user specified type an error is thrown.

* **function definition:**

        template <typename T> T get(const QString &key) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | Key to retrieve a value from [metadata](#file-members-m_metadata)

* **output:** (<tt>T</tt>) Returns a value of type <tt>T</tt>. <tt>T</tt> is a user specified type. The value associated with the given key must be convertable to <tt>T</tt>.
* **see:** [get](#file-function-get-2), [getList](#file-function-getlist-1)
* **example:**

        File f;
        f.set("Key1", QVariant::fromValue<float>(1));

        f.get<float>("Key1");  // returns 1
        f.get<float>("Key2");  // Error: Key2 is not in the metadata
        f.get<QRectF>("Key1"); // Error: A float can't be converted to a QRectF

### T get(const [QString][QString] &key, const T &defaultValue) {: #file-function-get-2 }

Get a value from the [metadata](#file-members-m_metadata) using a provided key. If the key does not exist or the value cannot be converted to user specified type a provided default value is returned instead.

* **function definition:**

        template <typename T> T get(const QString &key, const T &defaultValue)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | Key to retrieve a value from the [metadata](#file-members-m_metadata)
    defaultValue | const T & | Default value to be returned if the key does not exist or found value cannot be converted to <tt>T</tt>. <tt>T</tt> is a user specified type.

* **output:** (<tt>T</tt>) Returns a value of type <tt>T</tt>. <tt>T</tt> is a user specified type. If the value associated with the key is invalid, the provided default value is returned instead.
* **see:** [get](#file-function-get-1), [getList](#file-function-getlist-1)
* **example:**

        File f;
        f.set("Key1", QVariant::fromValue<float>(1));

        f.get<float>("Key1", 5);  // returns 1
        f.get<float>("Key2", 5);  // returns 5
        f.get<QRectF>("Key1", QRectF(0, 0, 10, 10)); // returns QRectF(0, 0, 10x10)


### bool getBool(const [QString][QString] &key, bool defaultValue = false) const {: #file-function-getbool }

Get a boolean value from the [metadata](#file-members-m_metadata) using a provided key. If the key is not in the [metadata](#file-members-m_metadata) a provided default value is returned. If the key is in the metadata but the value cannot be converted to a bool true is returned. If the key is found and the value can be converted to a bool the value is returned.

* **function definition:**

        bool getBool(const QString &key, bool defaultValue = false) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | Key to retrieve a value from the [metadata](#file-members-m_metadata)
    defaultValue | bool | (Optional) Default value to be returned if the key is not in the [metadata](#file-members-m_metadata).

* **output:** (bool) If the key *is not* in the [metadata](#file-members-m_metadata) the provided default value is returned. If the key *is* in the [metadata](#file-members-m_metadata) but the associated value *cannot* be converted to a bool true is returned. If the key *is* in the [metadata](#file-members-m_metadata) and the associated value *can* be converted to a bool, that value is returned.
* **see:** [get](#file-function-get-2)
* **example:**

        File f;
        f.set("Key1", QVariant::fromValue<bool>(true));
        f.set("Key2", QVariant::fromValue<float>(10));

        f.getBool("Key1");       // returns true
        f.getBool("Key2")        // returns true (key found)
        f.getBool("Key3");       // returns false (default value)
        f.getBool("Key3", true); // returns true (default value)


### [QList][QList]&lt;T&gt; getList(const [QString][QString] &key) const {: #file-function-getlist-1 }

Get a list from the [metadata](#file-members-m_metadata) using a provided key. If the key does not exist or the elements of the list cannot be converted to a user specified type an error is thrown.

* **function definition:**

        template <typename T> QList<T> getList(const QString &key) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | Key to retrieve a value from the [metadata](#file-members-m_metadata)

* **output:** ([QList][QList]&lt;<tt>T</tt>&gt;) Returns a list of values of a user specified type.
* **see:** [setList](#file-function-setlist), [get](#file-function-get-1)
* **example:**

        File file;

        QList<float> list = QList<float>() << 1 << 2 << 3;
        file.setList<float>("List", list);

        file.getList<float>("List");  // return [1., 2. 3.]
        file.getList<QRectF>("List"); // Error: float cannot be converted to QRectF
        file.getList<float>("Key");   // Error: key doesn't exist


### [QList][QList]&lt;T&gt; getList(const [QString][QString] &key, const [QList][QList]&lt;T&gt; defaultValue) const {: #file-function-getlist-2 }

Get a list from the [metadata](#file-members-m_metadata) using a provided key. If the key does not exist or the elements of the list cannot be converted to a user specified type a provided default value is returned.

* **function definition:**

template <typename T> QList<T> getList(const QString &key, const QList<T> defaultValue) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | Key to retrieve a value from the [metadata](#file-members-m_metadata)
    defaultValue | [QList][QList]&lt;<tt>T</tt> | (Optional) Default value to be returned if the key is not in the [metadata](#file-members-m_metadata).

* **output:** ([QList][QList]&lt;<tt>T</tt>&gt;) Returns a list of values of user specified type. If key is not in the [metadata](#file-members-m_metadata) or if the value cannot be converted to a [QList][QList]&lt;<tt>T</tt>&gt; the default value is returned.
* **see:** [getList](#file-function-getlist-1)
* **example:**

        File file;

        QList<float> list = QList<float>() << 1 << 2 << 3;
        file.setList<float>("List", list);

        file.getList<float>("List", QList<float>());                  // return [1., 2. 3.]
        file.getList<QRectF>("List", QList<QRectF>());                // return []
        file.getList<float>("Key", QList<float>() << 1 << 2 << 3);    // return [1., 2., 3.]


### [QList][QList]&lt;[QPointF][QPointF]&gt; namedPoints() const {: #file-function-namedpoints }

Find values in the [metadata](#file-members-m_metadata) that can be converted into [QPointF][QPointF]'s. Values stored as [QList][QList]&lt;[QPointF][QPointF]&gt; *will not** be returned.

* **function definition:**

        QList<QPointF> namedPoints() const

* **parameters:** NONE
* **output:** ([QList][QList]&lt;[QPointF][QPointF]&gt;) Returns a list of points that can be converted from [metadata](#file-members-m_metadata) values.
* **example:**

        File file;
        file.set("Key1", QVariant::fromValue<QPointF>(QPointF(1, 1)));
        file.set("Key2", QVariant::fromValue<QPointF>(QPointF(2, 2)));
        file.set("Points", QVariant::fromValue<QPointF>(QPointF(3, 3)))

        f.namedPoints(); // returns [QPointF(1, 1), QPointF(2, 2), QPointF(3, 3)]

        file.setPoints(QList<QPointF>() << QPointF(3, 3)); // changes metadata["Points"] to QList<QPointF>
        f.namedPoints(); // returns [QPointF(1, 1), QPointF(2, 2)]


### [QList][QList]&lt;[QPointF][QPointF]&gt; points() const {: #file-function-points }

Get values stored in the [metadata](#file-members-m_metadata) with key "Points". It is expected that this field holds a [QList][QList]&lt;[QPointf][QPointF]>&gt;.

* **function definition:**

        QList<QPointF> points() const

* **parameters:** NONE
* **output:** ([QList][QList]&lt;[QPointf][QPointF]>&gt;) Returns a list of points stored at [metadata](#file-members-m_metadata)["Points"]
* **see:** [appendPoint](#file-function-appendpoint), [appendPoints](#file-function-appendpoints), [clearPoints](#file-function-clearpoints), [setPoints](#file-function-setpoints)
* **example:**

        File file;
        file.set("Points", QVariant::fromValue<QPointF>(QPointF(1, 1)));
        file.points(); // returns [] (point is not in a list)

        file.setPoints(QList<QPointF>() << QPointF(2, 2));
        file.points(); // returns [QPointF(2, 2)]


### void appendPoint(const [QPointF][QPointF] &point) {: #file-function-appendpoint }

Append a point to the [QList][QList]&lt;[QPointF][QPointF]&gt; stored at [metadata](#file-members-m_metadata)["Points"].

* **function definition:**

        void appendPoint(const QPointF &point)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    point | const [QPoint][QPoint] & | Point to be appended

* **output:** (void)
* **example:**

        File file;
        file.points(); // returns []

        file.appendPoint(QPointF(1, 1));
        file.points(); // returns [QPointF(1, 1)]


### void appendPoints(const [QList][QList]&lt;[QPointF][QPointF]&gt; &points) {: #file-function-appendpoints }

Append a list of points to the [QList][QList]&lt;[QPointF][QPointF]&gt; stored at [metadata](#file-members-m_metadata)["Points"].

* **function definition:**

        void appendPoints(const QList<QPointF> &points)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    points | const [QList][QList]&lt;[QPointF][QPointF]&gt; & | List of points to be appended

* **output:** (void)
* **example:**

        File file;
        file.points(); // returns []

        file.appendPoints(QList<QPointF>() << QPointF(1, 1) << QPointF(2, 2));
        file.points(); // returns [QPointF(1, 1), QPointF(2, 2)]


### void clearPoints() {: #file-function-clearpoints }

Remove all points stored at [metadata](#file-members-m_metadata)["Points"].

* **function definition:**

        inline void clearPoints()

* **parameters:** NONE
* **output:** (void)
* **example:**

        File file;
        file.appendPoints(QList<QPointF>() << QPointF(1, 1) << QPointF(2, 2));
        file.points(); // returns [QPointF(1, 1), QPointF(2, 2)]

        file.clearPoints();
        file.points(); // returns []


### void setPoints(const [QList][QList]&lt;[QPointF][QPointF]&gt; &points) {: #file-function-setpoints }

Replace the points stored at [metadata](#file-members-m_metadata)["Points"]

* **function definition:**

        inline void setPoints(const QList<QPointF> &points)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    points | const [QList][QList]&lt;[QPointF][QPointF]&gt; & | Points to overwrite [metadata](#file-members-m_metadata) with

* **output:** (void)
* **example:**

        File file;
        file.appendPoints(QList<QPointF>() << QPointF(1, 1) << QPointF(2, 2));
        file.points(); // returns [QPointF(1, 1), QPointF(2, 2)]

        file.setPoints(QList<QPointF>() << QPointF(3, 3) << QPointF(4, 4));
        file.points(); // returns [QPointF(3, 3), QPointF(4, 4)]


### [QList][QList]&lt;[QRectF][QRectF]&gt; namedRects() const {: #file-function-namedrects }

Find values in the [metadata](#file-members-m_metadata) that can be converted into [QRectF][QRectF]'s. Values stored as [QList][QList]&lt;[QRectF][QRectF]&gt; *will not** be returned.

* **function definition:**

        QList<QRectF> namedRects() const

* **parameters:** NONE
* **output:** ([QList][QList]&lt;[QRectF][QRectF]&gt;) Returns a list of rects that can be converted from [metadata](#file-members-m_metadata) values.
* **example:**

        File file;
        file.set("Key1", QVariant::fromValue<QRectF>(QRectF(1, 1, 5, 5)));
        file.set("Key2", QVariant::fromValue<QRectF>(QRectF(2, 2, 5, 5)));
        file.set("Rects", QVariant::fromValue<QRectF>(QRectF(3, 3, 5, 5)));

        f.namedRects(); // returns [QRectF(1, 1, 5x5), QRectF(2, 2, 5x5), QRectF(3, 3, 5x5)]

        file.setRects(QList<QRectF>() << QRectF(3, 3, 5x5)); // changes metadata["Rects"] to QList<QRectF>
        f.namedRects(); // returns [QRectF(1, 1, 5x5), QRectF(2, 2, 5x5)]


### [QList][QList]&lt;[QRectF][QRectF]&gt; rects() const {: #file-function-rects }

Get values stored at [metadata](#file-members-m_metadata)["Rects"]. It is expected that this field holds a [QList][QList]&lt;[QRectf][QRectF]>&gt;.

* **function definition:**

        QList<QRectF> points() const

* **parameters:** NONE
* **output:** ([QList][QList]&lt;[QRectf][QRectF]>&gt;) Returns a list of rects stored at [metadata](#file-members-m_metadata)["Rects"]
* **see:** [appendRect](#file-function-appendrect-1), [appendRects](#file-function-appendrects-1), [clearRects](#file-function-clearrects), [setRects](#file-function-setrects-1)
* **example:**

        File file;
        file.set("Rects", QVariant::fromValue<QRectF>(QRectF(1, 1, 5, 5)));
        file.rects(); // returns [] (rect is not in a list)

        file.setRects(QList<QRectF>() << QRectF(2, 2, 5, 5));
        file.rects(); // returns [QRectF(2, 2, 5x5)]


### void appendRect(const [QRectF][QRectF] &rect) {: #file-function-appendrect-1 }

Append a rect to the [QList][QList]&lt;[QRectF][QRectF]&gt; stored at [metadata](#file-members-m_metadata)["Rects"].

* **function definition:**

        void appendRect(const QRectF &rect)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    rect | const [QRect][QRect] & | Rect to be appended

* **output:** (void)
* **example:**

        File file;
        file.rects(); // returns []

        file.appendRect(QRectF(1, 1, 5, 5));
        file.rects(); // returns [QRectF(1, 1, 5x5)]


### void appendRect(const [Rect][Rect] &rect) {: #file-function-appendrect-2 }

Append an OpenCV-style [Rect][Rect] to the [QList][QList]&lt;[QRectF][QRectF]&gt; stored at [metadata](#file-members-m_metadata)["Rects"]. Supplied OpenCV-style rects are converted to [QRectF][QRectF] before being appended.

* **function definition:**

        void appendRect(const Rect &rect)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    rect | const [Rect][Rect] & | OpenCV-style rect to be appended

* **output:** (void)
* **example:**

        File file;
        file.rects(); // returns []

        file.appendRect(cv::Rect(1, 1, 5, 5)); // automatically converted to QRectF
        file.rects(); // returns [QRectF(1, 1, 5x5)]


### void appendRects(const [QList][QList]&lt;[QRectF][QRectF]&gt; &rects) {: #file-function-appendrects-1 }

Append a list of rects to the [QList][QList]&lt;[QRectF][QRectF]&gt; stored at [metadata](#file-members-m_metadata)["Rects"].

* **function definition:**

        void appendRects(const QList<QRectF> &rects)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    rects | const [QList][QList]&lt;[QRectF][QRectF]&gt; & | List of rects to be appended

* **output:** (void)
* **example:**

        File file;
        file.rects(); // returns []

        file.appendRects(QList<QRectF>() << QRectF(1, 1, 5, 5) << QRectF(2, 2, 5, 5));
        file.rects(); // returns [QRectF(1, 1, 5x5), QRectF(2, 2, 5x5)]


### void appendRects(const [QList][QList]&lt;[QRectF][QRectF]&gt; &rects) {: #file-function-appendrects-2 }

Append a list of OpenCV-style [Rects][Rect] to the [QList][QList]&lt;[QRectF][QRectF]&gt; stored at [metadata](#file-members-m_metadata)["Rects"]. Supplied OpenCV-style rects are converted to [QRectF][QRectF] before being appended.

* **function definition:**

        void appendRects(const QList<QRectF> &rects)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    rects | const [QList][QList]&lt;[Rect][Rect]&gt; & | List of OpenCV-style rects to be appended

* **output:** (void)
* **example:**

        File file;
        file.rects(); // returns []

        file.appendRects(QList<Rect>() << Rect(1, 1, 5, 5) << Rect(2, 2, 5, 5));
        file.rects(); // returns [QRectF(1, 1, 5x5), QRectF(2, 2, 5x5)]

### void clearRects() {: #file-function-clearrects }

Remove all points stored at [metadata](#file-members-m_metadata)["Rects"].

* **function definition:**

        inline void clearRects()

* **parameters:** NONE
* **output:** (void)
* **example:**

        File file;
        file.appendRects(QList<QRectF>() << QRectF(1, 1, 5, 5) << QRectF(2, 2, 5, 5));
        file.rects(); // returns [QRectF(1, 1, 5x5), QRectF(2, 2, 5x5)]

        file.clearRects();
        file.rects(); // returns []


### void setRects(const [QList][QList]&lt;[QRectF][QRectF]&gt; &rects) {: #file-function-setrects-1 }

Replace the rects stored at [metadata](#file-members-m_metadata) with a provided list of rects.

* **function definition:**

        inline void setRects(const QList<QRectF> &rects)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    rects | const [QList][QList]&lt;[QRectF][QRectF]&gt; & | Rects to overwrite [metadata](#file-members-m_metadata)["Rects"] with

* **output:** (void)
* **example:**

        File file;
        file.appendRects(QList<QRectF>() << QRectF(1, 1, 5, 5) << QRectF(2, 2, 5, 5));
        file.rects(); // returns [QRectF(1, 1, 5x5), QRectF(2, 2, 5x5)]

        file.setRects(QList<QRectF>() << QRectF(3, 3, 5, 5) << QRectF(4, 4, 5, 5));
        file.rects(); // returns [QRectF(3, 3, 5x5), QRectF(4, 4, 5x5)]


### void setRects(const [QList][QList]&lt;[Rect][Rect]&gt; &rects) {: #file-function-setrects-2 }

Replace the rects stored at [metadata](#file-members-m_metadata) with a provided list of OpenCV-style [Rects][Rect].

* **function definition:**

        inline void setRects(const QList<Rect> &rects)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    rects | const [QList][QList]&lt;[Rect][Rect]&gt; & | OpenCV-style rects to overwrite [metadata](#file-members-m_metadata)["Rects"] with

* **output:** (void)
* **example:**

        File file;
        file.appendRects(QList<cv::Rect>() << cv::Rect(1, 1, 5, 5) << cv::Rect(2, 2, 5, 5));
        file.rects(); // returns [QRectF(1, 1, 5x5), QRectF(2, 2, 5x5)]

        file.setRects(QList<cv::Rect>() << cv::Rect(3, 3, 5, 5) << cv::Rect(4, 4, 5, 5));
        file.rects(); // returns [QRectF(3, 3, 5x5), QRectF(4, 4, 5x5)]
