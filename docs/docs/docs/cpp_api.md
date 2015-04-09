# API Functions

## IsClassifier {: #function-isclassiifer }

Determines if the given algorithm is a classifier. A classifier is defined as a [Transform](#transform) with no associated [Distance](#distance). Instead metadata fields with the predicted output classes are populated in [Template](#template) [files](#template-members-file).

* **function definition:**

        bool IsClassifier(const QString &algorithm)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    algorithm | const [QString][QString] & | Algorithm to evaluate

* **output:** (bool) True if the algorithm is a classifier and false otherwise
* **see:** [br_is_classifier](c_api.md#br_is_classifier)
* **example:**

        IsClassifier("Identity"); // returns true
        IsClassifier("Identity:Dist"); // returns false

## Train {: #function-train }

High level function for creating models.

* **function definition:**

        void Train(const File &input, const File &model)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    input | const [File](#file) & | Training data
    model | const [File](#file) & | Model file

* **output:** (void)
* **see:** The [training tutorial](../tutorials.md#training-algorithms) for an example of training.
* **example:**

        File file("/path/to/images/or/gallery.gal");
        File model("/path/to/model/file");
        Train(file, model);

## Enroll {: #function-enroll }

High level function for creating [galleries](#gallery).

* **function definition:**

        void Enroll(const File &input, const File &gallery = File())

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    input | const [File](#file) & | Path to enrollment file
    gallery | const [File](#file) & | (Optional) Path to gallery file.

* **output:** (void)
* **see:** [br_enroll](c_api.md#br_enroll)
* **example:**

        File file("/path/to/images/or/gallery.gal");
        Enroll(file); // Don't need to specify a gallery file
        File gallery("/path/to/gallery/file");
        Enroll(file, gallery); // Will write to the specified gallery file

## Enroll {: #function-enroll }

High level function for enrolling templates. Templates are modified in place as they are projected through the algorithm.

* **function definition:**

        void Enroll(TemplateList &tmpl)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    tmpl | [TemplateList](#templatelist) & | Data to enroll

* **output:** (void)
* **example:**

        TemplateList tList = TemplateList() << Template("picture1.jpg")
                                            << Template("picture2.jpg")
                                            << Template("picture3.jpg");
        Enroll(tList);

## Project {: #function-project}

A naive alternative to [Enroll](#function-enroll-1).

* **function definition:**

        void Project(const File &input, const File &output)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    input | const [File](#file) & | Path to enrollment file
    gallery | const [File](#file) & | Path to gallery file.

* **output:** (void)
* **see:** [Enroll](#function-enroll-1)
* **example:**

        File file("/path/to/images/or/gallery.gal");
        File output("/path/to/gallery/file");
        Project(file, gallery); // Will write to the specified gallery file

## Compare {: #function-compare }

High level function for comparing galleries. Each template in the **queryGallery** is compared against every template in the **targetGallery**.

* **function definition:**

        void Compare(const File &targetGallery, const File &queryGallery, const File &output)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    targetGallery | const [File](#file) & | Gallery of target templates
    queryGallery | const [File](#file) & | Gallery of query templates
    output | const [File](#file) & | Output file for results

* **returns:** (output)
* **see:** [br_compare](c_api.md#br_compare)
* **example:**

        File target("/path/to/target/images/");
        File query("/path/to/query/images/");
        File output("/path/to/output/file");
        Compare(target, query, output);

## CompareTemplateList {: #function-comparetemplatelists}

High level function for comparing templates.

* **function definition:**

        void CompareTemplateLists(const TemplateList &target, const TemplateList &query, Output *output);

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    target | const [TemplateList](#templatelist) & | Target templates
    query | const [TemplateList](#templatelist) & | Query templates
    output | [Output](#output) \* | Output file for results

* **output:** (void)
* **example:**

        TemplateList targets = TemplateList() << Template("target_img1.jpg")
                                              << Template("target_img2.jpg")
                                              << Template("target_img3.jpg");

        TemplateList query = TemplateList() << Template("query_img.jpg");
        Output *output = Factory::make<Output>("/path/to/output/file");

        CompareTemplateLists(targets, query, output);


## PairwiseCompare {: #function-pairwisecompare }

High level function for doing a series of pairwise comparisons.

* **function definition:**

        void PairwiseCompare(const File &targetGallery, const File &queryGallery, const File &output)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    targetGallery | const [File](#file) & | Gallery of target templates
    queryGallery | const [File](#file) & | Gallery of query templates
    output | const [File](#file) & | Output file for results  

* **output:** (void)
* **see:** [br_pairwise_comparison](c_api.md#br_pairwise_compare)
* **example:**

        File target("/path/to/target/images/");
        File query("/path/to/query/images/");
        File output("/path/to/output/file");
        PairwiseCompare(target, query, output);

## Convert {: #function-convert }

Change the format of the **inputFile** to the format of the **outputFile**. Both the **inputFile** format and the **outputFile** format must be of the same format group, which is specified by the **fileType**.

* **function definition:**

        void Convert(const File &fileType, const File &inputFile, const File &outputFile)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    fileType | const [File](#file) & | Can be either: <ul> <li>[Format](#format)</li> <li>[Gallery](#gallery)</li> <li>[Output](#output)</li> </ul>
    inputFile | const [File](#file) & | File to be converted. Format is inferred from the extension.
    outputFile | const [File](#file) & | File to store converted input. Format is inferred from the extension.

* **output:** (void)
* **example:**

        File input("input.csv");
        File output("output.xml");
        Convert("Format", input, output);

## Cat {: #function-cat }

Concatenate several galleries into one.

* **function definition:**

        void Cat(const QStringList &inputGalleries, const QString &outputGallery)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    inputGalleries | const [QStringList][QStringList] & | List of galleries to concatenate
    outputGallery | const [QString][QString] & | Gallery to store the concatenated result. This gallery cannot be in the inputGalleries

* **output:** (void)
* **see:** [br_cat](c_api.md#br_cat)
* **example:**

        QStringList inputGalleries = QStringList() << "/path/to/gallery1"
                                                   << "/path/to/gallery2"
                                                   << "/path/to/gallery3";

        QString outputGallery = "/path/to/outputGallery";
        Cat(inputGalleries, outputGallery);

## Deduplicate {: #function-deduplicate }

Deduplicate a gallery. A duplicate is defined as an image with a match score above a given threshold to another image in the gallery.

* **function definition:**

        void Deduplicate(const File &inputGallery, const File &outputGallery, const QString &threshold)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    inputGallery | const [File](#file) & | Gallery to deduplicate
    outputGallery | const [File](#file) & | Gallery to store the deduplicated result
    threshold | const [QString][QString] & | Match score threshold to determine duplicates

* **output:** (void)
* **see:** [br_deduplicate](c_api.md#br_deduplicate)
* **example:**

        File input("/path/to/input/galley/with/dups");
        File output("/path/to/output/gallery");
        Deduplicate(input, output, "0.7"); // Remove duplicates with match scores above 0.7

---

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

---

# FileList

Inherits [QList][QList]&lt;[File](#file)&gt;.

A convenience class for dealing with lists of files.

## Members {: #filelist-members }

NONE

---

## Constructors {: #filelist-constructors }

Constructor | Description
--- | ---
FileList() | Default constructor. Doesn't do anything.
FileList(int n) | Intialize the [FileList](#filelist) with n empty [Files](#file)
FileList(const [QStringList][QStringList] &files) | Initialize the [FileList](#filelist) with a list of strings. Each string should have the format "filename[key1=value1, key2=value2, ... keyN=valueN]"
FileList(const [QList][QList]&lt;[File](#file)&gt; &files) | Initialize the [FileList](#filelist) from a list of [files](#file).

---

## Static Functions {: #filelist-static-functions }


### [FileList](#filelist) fromGallery(const [File](#file) &gallery, bool cache = false) {: #filelist-static-fromgallery }

Create a [FileList](#filelist) from a [Gallery](#gallery). Galleries store one or more [Templates](#template) on disk. Common formats include **csv**, **xml**, and **gal**, which is a unique OpenBR format. Read more about this in the [Gallery](#gallery) section. This function creates a [FileList](#filelist) by parsing the stored gallery based on its format. Cache determines whether the gallery should be stored for faster reading later.

* **function definition:**

        static FileList fromGallery(const File &gallery, bool cache = false)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    gallery | const [File](#file) & | Gallery file to be enrolled
    cache | bool | (Optional) Retain the gallery in memory. Default is false.

* **output:** ([FileList](#filelist)) Returns the filelist that the gallery was enrolled into
* **example:**

        File gallery("gallery.csv");

        FileList fList = FileList::fromGallery(gallery);
        fList.flat(); // returns all the files that have been loaded from disk. It could
                      // be 1 or 100 depending on what was stored.

---

## Functions {: #filelist-functions }


### [QStringList][QStringList] flat() const {: #filelist-function-flat }

Calls [flat](#file-function-flat) on every [File](#file) in the list and returns the resulting strings as a [QStringList][QStringList].

* **function definition:**

        QStringList flat() const

* **parameters:** NONE
* **output:** ([QStringList][QStringList]) Returns a list of the output of calling [flat](#file-function-flat) on each [File](#file)
* **example:**

        File f1("picture1.jpg"), f2("picture2.jpg");
        f1.set("Key", QString("Value"));

        FileList fList(QList<File>() << f1 << f2);
        fList.flat(); // returns ["picture1.jpg[Key=Value]", "picture2.jpg"]


### [QStringList][QStringList] names() const {: #filelist-function-names }

Get the [names](#file-members-name) of every [File](#file) in the list.

* **function definition:**

        QStringList names() const

* **parameters:** NONE
* **output:** ([QStringList][QStringList]) Returns the [name](#file-members-name) of every [File](#file) in the list
* **example:**

        File f1("picture1.jpg"), f2("picture2.jpg");
        f1.set("Key", QString("Value"));

        FileList fList(QList<File>() << f1 << f2);
        fList.names(); // returns ["picture1.jpg", "picture2.jpg"]


### void sort(const [QString][QString] &key) {: #filelist-function-sort }

Sort the [FileList](#filelist) based on the values associated with a provided key in each [File](#file).

* **function definition:**

        void sort(const QString &key)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | Key to look up desired values in each [Files](#file) [metadata](#file-members-m_metadata)

* **output:** (void)
* **example:**

        File f1("1"), f2("2"), f3("3");
        f1.set("Key", QVariant::fromValue<float>(3));
        f2.set("Key", QVariant::fromValue<float>(1));
        f3.set("Key", QVariant::fromValue<float>(2));

        FileList fList(QList<File>() << f1 << f2 << f3);
        fList.names(); // returns ["1", "2", "3"]

        fList.sort("Key");
        fList.names(); // returns ["2", "3", "1"]


### [QList][QList]&lt;int&gt; crossValidationPartitions() const {: #filelist-function-crossvalidationpartitions }

Get the cross-validation partion of each [File](#file) in the list. The partition is stored in each [File](#file) at [metadata](#file-members-m_metadata)["Partition"].

* **function definition:**

        QList<int> crossValidationPartitions() const

* **parameters:** NONE
* **output:** ([QList][QList]&lt;int&gt;) Returns the cross-validation partion of each [File](#file) as a list. If a [File](#file) does not have the "Partition" field in it's [metadata](#file-members-m_metadata) 0 is used.
* **example:**

        File f1, f2, f3;
        f1.set("Partition", QVariant::fromValue<int>(1));
        f3.set("Partition", QVariant::fromValue<int>(3));

        FileList fList(QList<File>() << f1 << f2 << f3);
        fList.crossValidationPartitions(); // returns [1, 0, 3]


### int failures() const {: #filelist-function-failures }

Get the number of [Files](#file) in the list that have [failed to enroll](#file-members-fte).

* **function definition:**

        int failures() const

* **parameters:** NONE
* **output:** (int) Returns the number of [Files](#file) that have [fte](#file-members-fte) equal true.
* **example:**

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

Metadata related to the template that is computed during enrollment (ex. bounding boxes, eye locations, quality metrics, ...) should be assigned to the template's [File](#template-members-file) member.

## Members {: #template-members }

Member | Type | Description
--- | --- | ---
<a class="table-anchor" id=template-members-file></a>file | [File](#file) | The file that constructs the template and stores its associated metadata

---

## Constructors {: #template-constructors }

Constructor | Description
--- | ---
Template() | The default template constructor. It doesn't do anything.
Template(const [File](#file) &file) | Sets [file](#template-members-file) to the given [File](#file).
Template(const [File](#file) &file, const [Mat][Mat] &mat) | Sets [file](#template-members-file) to the given [File](#file) and appends the given [Mat][Mat] to itself.
Template(const [Mat][Mat] &mat) | Appends the given [Mat][Mat] to itself

---

## Static Functions {: #template-static-functions }


### [QDataStream][QDataStream] &operator<<([QDataStream][QDataStream] &stream, const [Template](#template) &t) {: #template-static-operator-ltlt }

Serialize a template

* **function definition:**

        QDataStream &operator<<(QDataStream &stream, const Template &t)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    stream | [QDataStream][QDataStream] & | The stream to serialize to
    t | const [Template](#template) & | The template to serialize

* **output:** ([QDataStream][QDataStream] &) Returns the updated stream
* **example:**

        void store(QDataStream &stream)
        {
            Template t("picture.jpg");
            t.append(Mat::ones(1, 1, CV_8U));

            stream << t; // "["1"]picture.jpg" serialized to the stream
        }

### [QDataStream][QDataStream] &operator>>([QDataStream][QDataStream] &stream, [Template](#template) &t) {: #template-static-operator-gtgt }

Deserialize a template

* **function definition:**

        QDataStream &operator>>(QDataStream &stream, Template &t)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    stream | [QDataStream][QDataStream] & | The stream to deserialize to
    t | const [Template](#template) & | The template to deserialize

* **output:** ([QDataStream][QDataStream] &) Returns the updated stream
* **example:**

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

## Functions {: #template-functions }


### operator const [File](#file) &() const {: #template-function-operator-file }

Idiom to treat the template like a [File](#file).

* **function definition:**

        inline operator const File &() const

* **parameters:** NONE
* **output:** ([File](#file) Returns [file](#template-members-file).


### const [Mat][Mat] &m() const {: #template-function-m-1 }

Idiom to treat the template like a [Mat][Mat].

* **function definition:**

        inline const Mat &m() const

* **parameters:** NONE
* **output:** ([Mat][Mat]) Returns the last [Mat][Mat] in the list. If the list is empty an empty [Mat][Mat] is returned.
* **example:**

        Template t;
        t.m(); // returns empty mat

        Mat m1;
        t.append(m1);
        t.m(); // returns m1;

        Mat m2;
        t.append(m2);
        t.m(); // returns m2;


### [Mat][Mat] &m() {: #template-function-m-2 }

Idiom to treat the template like a [Mat][Mat].

* **function definition:**

        inline Mat &m()

* **parameters:** NONE
* **output:** ([Mat][Mat]) Returns the last [Mat][Mat] in the list. If the list is empty an empty [Mat][Mat] is returned.
* **example:**

        Template t;
        t.m(); // returns empty mat

        Mat m1;
        t.append(m1);
        t.m(); // returns m1;

        Mat m2;
        t.append(m2);
        t.m(); // returns m2;


### operator const [Mat][Mat] &() {: #template-function-operator-mat-1 }

Idiom to treat the template like a [Mat][Mat]. Makes a call to [m()](#template-function-m-1).

* **function definition:**

        inline operator const Mat&() const

* **parameters:** NONE
* **output:** ([Mat][Mat]) Returns the last [Mat][Mat] in the list. If the list is empty an empty [Mat][Mat] is returned.
* **see:** [m](#template-function-m-1)


### operator [Mat][Mat] &() {: #template-function-operator-mat-2 }

Idiom to treat the template like a [Mat][Mat]. Makes a call to [m()](#template-function-m-1).

* **function definition:**

        inline operator Mat&()

* **parameters:** NONE
* **output:** ([Mat][Mat]) Returns the last [Mat][Mat] in the list. If the list is empty an empty [Mat][Mat] is returned.
* **see:** [m](#template-function-m-1)


### operator [_InputArray][InputArray] &() {: #template-function-operator-inputarray }

Idiom to treat the template like an [_InputArray][InputArray]. Makes a call to [m()](#template-function-m-1).

* **function definition:**

        inline operator _InputArray() const

<!-- _no italics_-->
* **parameters:** NONE
* **output:** ([_InputArray][InputArray]) Returns the last [Mat][Mat] in the list. If the list is empty an empty [Mat][Mat] is returned.
* **see:** [m](#template-function-m-1)


### operator [_OutputArray][OutputArray] &() {: #template-function-operator-outputarray }

Idiom to treat the template like an [_OutputArray][InputArray]. Makes a call to [m()](#template-function-m-1).

* **function definition:**

        inline operator _OutputArray()

<!-- _no italics_-->
* **parameters:** NONE
* **output:** ([_OutputArray][OutputArray]) Returns the last [Mat][Mat] in the list. If the list is empty an empty [Mat][Mat] is returned.
* **see:** [m](#template-function-m-1)


### [Mat][Mat] &operator =(const [Mat][Mat] &other) {: #template-function-operator-e }

Idiom to treat the template like a [Mat][Mat]. Set the result of [m()](#template-function-m-1) equal to other.

* **function definition:**

        inline Mat &operator=(const Mat &other)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    other | const Mat & | Mat to overwrite value of [m](#template-function-m-1)

* **output**: ([Mat][Mat] &) Returns a reference to the updated [Mat][Mat]


### bool isNull() const {: #template-function-isnull }

Check if the template is NULL.

* **function definition:**

        inline bool isNull() const

* **parameters:** NONE
* **output:** (bool) Returns true if the template is empty *or* if [m](#template-function-m-1) has no data.
* **example:**

        Template t;
        t.isNull(); // returns true

        t.append(Mat());
        t.isNull(); // returns true

        t.append(Mat::ones(1, 1, CV_8U));
        t.isNull(); // returns false


### void merge(const [Template](#template) &other) {: #template-function-merge }

Append the contents of another template. The [files](#template-members-file) are appended using [append](#file-function-append-1).

* **function definition:**

        inline void merge(const Template &other)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    other | const [Template][#template] & | Template to be merged

* **output:** (void)
* **example:**

        Template t1("picture1.jpg"), t2("picture2.jpg");
        Mat m1, m2;

        t1.append(m1);
        t2.append(m2);

        t1.merge(t2);

        t1.file; // returns picture1.jpg;picture2.jpg[seperator=;]
        t1; // returns [m1, m2]


### size_t bytes() const {: #template-function-bytes }

Get the total number of bytes in the template

* **function definition:**

        size_t bytes() const

* **parameters:** None
* **output:** (int) Returns the sum of the bytes in each [Mat][Mat] in the [Template](#template)
* **example:**

        Template t;

        Mat m1 = Mat::ones(1, 1, CV_8U); // 1 byte
        Mat m2 = Mat::ones(2, 2, CV_8U); // 4 bytes
        Mat m3 = Mat::ones(3, 3, CV_8U); // 9 bytes

        t << m1 << m2 << m3;

        t.bytes(); // returns 14


### Template clone() const {: #template-function-clone }

Clone the template

* **function definition:**

        Template clone() const

* **parameters:** NONE
* **output:** ([Template](#template)) Returns a new [Template](#template) with copies of the [file](#template-members-file) and each [Mat][Mat] that was in the original.
* **example:**

        Template t1("picture.jpg");
        t1.append(Mat::ones(1, 1, CV_8U));

        Template t2 = t1.clone();

        t2.file; // returns "picture.jpg"
        t2; // returns ["1"]

---

# TemplateList

Inherits [QList][QList]&lt;[Template][#template]&gt;.

Convenience class for working with a list of templates.

## Members {: #templatelist-members }

NONE

---

## Constructors {: #templatelist-constructors }

Constructor | Description
--- | ---
TemplateList() | The default [TemplateList](#templatelist) constructor. Doesn't do anything.
TemplateList(const [QList][QList]&lt;[Template](#template)&gt; &templates) | Initialize the [TemplateList](#templatelist) with a list of templates. The given list is appended
TemplateList(const [QList][QList]&lt;[File](#file)&gt; &files) | Initialize the [TemplateList](#templatelist) with a list of [Files](#file). Each [File](#file) is treated like a template and appended.

---

## Static Functions {: #templatelist-static-functions }

### [TemplateList](#templatelist) fromGallery(const [File](#file) &gallery) {: #templatelist-static-fromgallery }

Create a [TemplateList](#templatelist) from a gallery [File](#file).

* **function definition:**

        static TemplateList fromGallery(const File &gallery)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    gallery | const [File](#file) & | Gallery [file](#file) to be enrolled.

* **output:** ([TemplateList](#templatelist)) Returns a [TemplateList](#templatelist) created by enrolling the gallery.


### [TemplateList](#templatelist) fromBuffer(const [QByteArray][QByteArray] &buffer) {: #templatelist-static-frombuffer }

Create a template from a memory buffer of individual templates. This is compatible with **.gal** galleries.

* **function definition:**

        static TemplateList fromBuffer(const QByteArray &buffer)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    buffer | const [QByteArray][QByteArray] & | Raw data buffer to be enrolled

* **output:** ([TemplateList][TemplateList]) Returns a [TemplateList](#templatelist) created by enrolling the buffer


### [TemplateList](#templatelist) relabel(const [TemplateList](#templatelist) &tl, const [QString][QString] &propName, bool preserveIntegers) {: #templatelist-static-relabel }

Relabel the values associated with a given key in the [metadata](#file-members-m_metadata) of a provided [TemplateList](#templatelist). The values are relabeled to be between [0, numClasses-1]. If preserveIntegers is true and the [metadata](#file-members-m_metadata) can be converted to integers then numClasses equals the maximum value in the values. Otherwise, numClasses equals the number of unique values. The relabeled values are stored in the "Label" field of the returned [TemplateList](#templatelist).

* **function definition:**

        static TemplateList relabel(const TemplateList &tl, const QString &propName, bool preserveIntegers)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    tl | const [TemplateList](#templatelist) & | [TemplateList](#templatelist) to be relabeled
    propName | const [QString][QString] & | [Metadata](#file-members-m_metadata) key
    preserveIntegers | bool | If true attempt to use the [metadata](#file-members-m_metadata) values as the relabeled values. Otherwise use the number of unique values.

* **output:** ([TemplateList](#templatelist)) Returns a [TemplateList](#templatelist) identical to the input [TemplateList](#templatelist) but with the new labels appended to the [metadata](#file-members-m_metadata) using the "Label" key.
* **example:**

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

## Functions {: #templatelist-functions }

### [QList][QList]&lt;int&gt; indexProperty(const [QString][QString] &propName, [QHash][QHash]&lt;[QString][QString], int&gt; &valueMap, [QHash][QHash]&lt;int, [QVariant][QVariant]&gt; &reverseLookup) const {: #templatelist-function-indexproperty-1 }

Convert [metadata](#file-members-m_metadata) values associated with **propName** to integers. Each unique value gets its own integer. This is useful in many classification problems where nominal data (e.g "Male", "Female") needs to represented with integers ("Male" = 0, "Female" = 1). **valueMap** and **reverseLookup** are created to allow easy conversion to the integer replacements and back.

* **function definition:**

        QList<int> indexProperty(const QString &propName, QHash<QString, int> &valueMap, QHash<int, QVariant> &reverseLookup) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    propName | const [QString][QString] & | [Metadata](#file-members-m_metadata) key
    valueMap | [QHash][QHash]&lt;[QString][QString], int&gt; & | A mapping from [metadata](#file-members-m_metadata) values to the equivalent unique index. [QStrings][QString] are used instead of [QVariant][QVariant] so comparison operators can be used. This is filled in by the function and can be provided empty.
    reverseLookup | [QHash][QHash]&lt;int, [QVariant][QVariant]&gt; & | A mapping from the unique index to the original value. This is the *reverse* mapping of the **valueMap**. This is filled in by the function and can be provided empty.

* **output:** ([QList][QList]&lt;int&gt;) Returns a list of unique integers that can be mapped to the [metadata](#file-members-m_metadata) values associated with **propName**. The integers can be mapped to their respective values using **valueMap** and the values can be mapped to the integers using **reverseLookup**.
* **example:**

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


### [QList][QList]&lt;int&gt; indexProperty(const [QString][QString] &propName, [QHash][QHash]&lt;[QString][QString], int&gt; \*valueMap=NULL, [QHash][QHash]&lt;int, [QVariant][QVariant]&gt; \*reverseLookup=NULL) const {: #templatelist-function-indexproperty-2 }

Shortcut to call [indexProperty](#templatelist-function-indexproperty-1) without **valueMap** or **reverseLookup** arguments.

* **function definition:**

        QList<int> indexProperty(const QString &propName, QHash<QString, int> * valueMap=NULL,QHash<int, QVariant> * reverseLookup = NULL) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    propName | const [QString][QString] & | [Metadata](#file-members-m_metadata) key
    valueMap | [QHash][QHash]&lt;[QString][QString], int&gt; \* | (Optional) A mapping from [metadata](#file-members-m_metadata) values to the equivalent unique index. [QStrings][QString] are used instead of [QVariant][QVariant] so comparison operators can be used. This is filled in by the function and can be provided empty.
    reverseLookup | [QHash][QHash]&lt;int, [QVariant][QVariant]&gt; \* | (Optional) A mapping from the unique index to the original value. This is the *reverse* mapping of the **valueMap**. This is filled in by the function and can be provided empty.

* **output:** ([QList][QList]&lt;int&gt;) Returns a list of unique integers that can be mapped to the [metadata](#file-members-m_metadata) values associated with **propName**. The integers can be mapped to their respective values using **valueMap** (if provided) and the values can be mapped to the integers using **reverseLookup** (if provided).


### [QList][QList]&lt;int&gt; applyIndex(const [QString][QString] &propName, const [QHash][QHash]&lt;[QString][QString], int&gt; &valueMap) const {: #templatelist-function-applyindex }

Apply a mapping to convert non-integer values to integers. [Metadata](#file-members-m_metadata) values associated with **propName** are mapped through the given **valueMap**.

* **function definition:**

        QList<int> applyIndex(const QString &propName, const QHash<QString, int> &valueMap) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    propName | const [QString][QString] & | [Metadata](#file-members-m_metadata) key
    valueMap | const [QHash][QHash]&lt;[QString][QString], int&gt; & | (Optional) A mapping from [metadata](#file-members-m_metadata) values to the equivalent unique index. [QStrings][QString] are used instead of [QVariant][QVariant] so comparison operators can be used.

* **output:** ([Qlist][QList]&lt;int&gt;) Returns a list of integer values. The values are ordered in the same order as the [Templates](#template) in the list. The values are calculated like so:

    1. If the value *is* found in the **valueMap**, its integer mapping is appened to the list.
    2. If the value *is not* found in the **valueMap**, -1 is appened to the list.

* **example:**

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


### <tt>T</tt> bytes() const {: #templatelist-function-bytes }

Get the total number of bytes in the [TemplateList](#templatelist).

* **function definition:**

        template <typename T> T bytes() const

* **parameters:** NONE
* **output:** (<tt>T</tt>) Returns the sum of the bytes in each of the [Templates](#template) in the list. <tt>T</tt> is a user specified type. It is expected to be numeric (int, float etc.)
* **see:** [bytes](#template-function-bytes)
* **example:**

        Template t1, t2;

        t1.append(Mat::ones(1, 1, CV_8U)); // 1 byte
        t1.append(Mat::ones(2, 2, CV_8U)); // 4 bytes
        t2.append(Mat::ones(3, 3, CV_8U)); // 9 bytes
        t2.append(Mat::ones(4, 4, CV_8U)); // 16 bytes

        TemplateList tList(QList<Template>() << t1 << t2);
        tList.bytes(); // returns 30


### [QList][QList]&lt;[Mat][Mat]&gt; data(int index = 0) const {: #templatelist-function-data }

Get a list of matrices compiled from each [Template](#template) in the list.

* **function definition:**

        QList<cv::Mat> data(int index = 0) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    index | int | (Optional) Index into each [Template](#template) to select a [Mat][Mat]. Default is 0.

* **output:** ([QList][QList]&lt;[Mat][Mat]&gt;) Returns a list of [Mats][Mat]. One [Mat][Mat] is supplied by each [Template](#template) in the image at the specified index.
* **example:**

        Template t1, t2;

        t1.append(Mat::ones(1, 1, CV_8U));
        t1.append(Mat::zeros(1, 1, CV_8U));
        t2.append(Mat::ones(1, 1, CV_8U));
        t2.append(Mat::zeros(1, 1, CV_8U));

        TemplateList tList(QList<Template>() << t1 << t2);
        tList.data(); // returns ["1", "1"];
        tList.data(1); // returns ["0", "0"];


### [QList][QList]&lt;[TemplateList](#templatelist)&gt; partition(const [QList][QList]&lt;int&gt; &partitionSizes) const {: #templatelist-function-partition }

Divide the [TemplateList](#templatelist) into a list of [TemplateLists](#templatelist) partitions.

 * **function defintion:**

        QList<TemplateList> partition(const QList<int> &partitionSizes) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    partitionSizes | [QList][QList]&lt;int&gt; | A list of sizes for the partitions. The total number of partitions is equal to the length of this list. Each value in this list specifies the number of [Mats][Mat] that should be in each template of the associated partition. The sum of values in this list *must* equal the number of [Mats][Mat] in each [Template](#template) in the original [TemplateList](#templatelist).

* **output:** ([QList][QList]&lt;[TemplateList](#templatelist)&gt;) Returns a [QList][QList] of [TemplateLists](#templatelist) of partitions. Each partition has length equal to the number of templates in the original [TemplateList](#templatelist). Each [Template](#template) has length equal to the size specified in the associated value in **partitionSizes**.
* **example:**

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

### [FileList](#filelist) files() const {: #templatelist-function-files }

Get a list of all the [Files](#file) in the [TemplateList](#templatelist)

* **function definition:**

        FileList files() const

* **parameters:** NONE
* **output:** ([FileList](#filelist)) Returns a [FileList](#filelist) with the [file](#template-members-file) of each [Template](#template) in the [TemplateList](#templatelist).
* **example:**

        Template t1("picture1.jpg"), t2("picture2.jpg");

        t1.file.set("Key", QVariant::fromValue<float>(1));
        t2.file.set("Key", QVariant::fromValue<float>(2));

        TemplateList tList(QList<Template>() << t1 << t2);

        tList.files(); // returns ["picture1.jpg[Key=1]", "picture2.jpg[Key=2]"]


### [FileList](#filelist) operator()() {: #templatelist-function-operator-pp }

Shortcut call to [files](#templatelist-function-files)

* **function definition:**

        FileList operator()() const

* **parameters:** NONE
* **output:** ([FileList](#filelist)) Returns a [FileList](#filelist) with the [file](#template-members-file) of each [Template](#template) in the [TemplateList](#templatelist).
* **example:**

        Template t1("picture1.jpg"), t2("picture2.jpg");

        t1.file.set("Key", QVariant::fromValue<float>(1));
        t2.file.set("Key", QVariant::fromValue<float>(2));

        TemplateList tList(QList<Template>() << t1 << t2);

        tList.files(); // returns ["picture1.jpg[Key=1]", "picture2.jpg[Key=2]"]


### [QMap][QMap]&lt;T, int&gt; countValues(const [QString][QString] &propName, bool excludeFailures = false) const {: #templatelist-function-countvalues }

Get the frequency of each unique value associated with a provided [metadata](#file-members-m_metadata) key.

* **function definition:**

template<typename T> QMap<T,int> countValues(const QString &propName, bool excludeFailures = false) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    propName | const [QString][QString] & | [Metadata](#file-members-m_metadata) key
    excludeFailures | bool | (Optional) Exclude [File](#file) [metadata](#file-members-m_metadata) if the [File](#file) has [fte](#file-members-fte) equal to true. Default is false

* **output:** ([QMap][QMap]&lt;<T, int&gt;) Returns a mapping between unique [metadata](#file-members-m_metadata) and their frequency.
* **example:**

        Template t1, t2, t3, t4;

        t1.file.set("Key", QString("Class 1"));
        t2.file.set("Key", QString("Class 2"));
        t3.file.set("Key", QString("Class 3"));
        t4.file.set("Key", QString("Class 1"));

        TemplateList tList(QList<Template>() << t1 << t2 << t3 << t4);

        tList.countValues<QString>("Key"); // returns QMap(("Class 1", 2), ("Class 2", 1), ("Class 3", 1))

### [TemplateList](#templatelist) reduced() const {: #templatelist-function-reduced }

Reduce the [Templates](#template) in the [TemplateList](#templatelist) by merging them together.

* **function definition:**

        TemplateList reduced() const

* **parameters:** NONE
* **output:** ([TemplateList](#templatelist)) Returns a [TemplateList](#templatelist) with a single [Template](#template). The [Template](#template) is the result of calling [merge](#template-function-merge) on every [Template](#template).
* **see:** [merge](#template-function-merge)
* **example:**

        Template t1("picture1.jpg"), t2("picture2.jpg");

        t1.file.set("Key1", QString("Value1"));
        t2.file.set("Key2", QString("Value2"));

        TemplateList tList(QList<Template>() << t1 << t2);

        TemplateList reduced = tList.reduced();
        reduced.size(); // returns 1
        reduced.files(); // returns ["picture1.jpg;picture2.jpg[Key1=Value1, Key2=Value2, separator=;]"]

### [QList][QList]&lt;int&gt; find(const [QString][QString] &key, const T &value) {: #templatelist-function-find }

Get the indices of every [Template](#template) that has a provided key value pairing in its [metadata](#file-members-m_metadata)

* **function definition:**

        template<typename T> QList<int> find(const QString &key, const <tt>T</tt> &value)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | [Metadata](#file-members-m_metadata) key to search for
    value | const <tt>T</tt> & | Value to search for. Both the **key** and value must match. <tt>T</tt> is a user specified type.

* **output:** ([QList][QList]&lt;int&gt;) Returns a list of indices for [Templates](#template) that contained the key-value pairing in their [metadata](#file-members-m_metadata)
* **example:**

        Template t1, t2, t3;

        t1.file.set("Key", QString("Value1"));
        t2.file.set("Key", QString("Value2"));
        t3.file.set("Key", QString("Value2"));

        TemplateList tList(QList<Template>() << t1 << t2 << t3);
        tList.find<QString>("Key", "Value2"); // returns [1, 2]

---

# Factory

For run time construction of objects from strings.

Uses the Industrial Strength Pluggable Factory model described [here](http://adtmag.com/articles/2000/09/25/industrial-strength-pluggable-factories.aspx).

The factory might be the most important construct in OpenBR. It is what enables the entire plugin architecture to function. All plugins in OpenBR are children of the [Object](#object) type. At compile time, each plugin is registered in the factory using the plugin name and abstraction type ([Transform](#transform), [Distance](#distance), etc.). At runtime, OpenBR utilizes

## Members {: #factory-members }

Member | Type | Description
--- | --- | ---
registry | static [QMap][QMap]&lt;[QString][QString],[Factory](#factory)&lt;<tt>T</tt>&gt;\*&gt; | Registered

---

# Object

---

# Context

The singleton class of global settings. Before including and using OpenBR in a project the user must call [initialize](#context-static-initialize). Before the program terminates the user must call [finalize](#context-static-finalize). The settings are accessible as Context \*Globals.

## Members {: #context-members }

Member | Type | Description
--- | --- | ---
<a class="table-anchor" id=context-members-sdkpath></a>sdkPath | [QString][QString] | Path to the sdk. Path + **share/openbr/openbr.bib** must exist.
<a class="table-anchor" id=context-members-algorithm></a>algorithm | [QString][QString] | The default algorithm to use when enrolling and comparing templates.
<a class="table-anchor" id=context-members-log></a>log | [QString][QString] | Optional log file to copy **stderr** to.
<a class="table-anchor" id=context-members-path></a>path | [QString][QString] | Path to use when resolving images specified with relative paths. Multiple paths can be specified using a semicolon separator.
<a class="table-anchor" id=context-members-parallelism></a>parallelism | int | The number of threads to use. The default is the maximum of 1 and the value returned by ([QThread][QThread]::idealThreadCount() + 1).
<a class="table-anchor" id=context-members-usegui></a>useGui | bool | Whether or not to use GUI functions. The default is true.
<a class="table-anchor" id=context-members-blocksize></a>blockSize | int | The maximum number of templates to process in parallel. The default is: ```parallelism * ((sizeof(void*) == 4) ? 128 : 1024)```
<a class="table-anchor" id=context-members-quiet></a>quiet | bool | If true, no messages will be sent to the terminal. The default is false.
<a class="table-anchor" id=context-members-verbose></a>verbose | bool | If true, extra messages will be sent to the terminal. The default is false.
<a class="table-anchor" id=context-members-mostrecentmessage></a>mostRecentMessage | [QString][QString] | The most recent message sent to the terminal.
<a class="table-anchor" id=context-members-currentstep></a>currentStep | double | Used internally to compute [progress](#context-function-progress) and [timeRemaining](#context-function-timeremaining).
<a class="table-anchor" id=context-members-totalsteps></a>totalSteps | double | Used internally to compute [progress](#context-function-progress) and [timeRemaining](#context-function-timeremaining).
<a class="table-anchor" id=context-members-enrollall></a>enrollAll | bool | If true, enroll 0 or more templates per image. Otherwise, enroll exactly one. The default is false.
<a class="table-anchor" id=context-members-filters></a>filters | Filters | Filters is a ```typedef QHash<QString,QStringList> Filters```. Filters that automatically determine imposter matches based on target ([gallery](#gallery)) template metadata. See [FilterDistance](plugins/distance.md#filterdistance).
<a class="table-anchor" id=context-members-buffer></a>buffer | [QByteArray][QByteArray] | File output is redirected here if the file's basename is "buffer". This clears previous contents.
<a class="table-anchor" id=context-members-scorenormalization></a>scoreNormalization | bool | If true, enable score normalization. Otherwise disable it. The default is true.
<a class="table-anchor" id=context-members-crossValidate></a>crossValidate | int | Perform k-fold cross validation where k is the value of **crossValidate**. The default value is 0.
<a class="table-anchor" id=context-members-modelsearch></a>modelSearch | [QList][QList]&lt;[QString][QString]&gt; | List of paths to search for sub-models on.
<a class="table-anchor" id=context-members-abbreviations></a>abbreviations | [QHash][QHash]&lt;[QString][QString], [QString][QString]&gt; | Used by [Transform](#transform)::[make](#transform-function-make) to expand abbreviated algorithms into their complete definitions.
<a class="table-anchor" id=context-members-starttime></a>startTime | [QTime][QTime] | Used to estimate [timeRemaining](#context-function-timeremaining).
<a class="table-anchor" id=context-members-logfile></a>logFile | [QFile][QFile] | Log file to write to.

---

## Constructors {: #context-constructors }

NONE

---

## Static Functions {: #context-static-functions }

### void initialize(int &argc, char \*argv[], [QString][QString] sdkPath = "", bool useGui = true) {: #context-static-initialize }

Call *once* at the start of the application to allocate global variables. If the project is a [Qt][Qt] project this call should occur after initializing <tt>QApplication</tt>.

* **function definition:**

        static void initialize(int &argc, char *argv[], QString sdkPath = "", bool useGui = true);

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    argc | int & | Number of command line arguments as provided by <tt>main()</tt>
    argv | char * [] | Command line arguments as provided by <tt>main()</tt>
    sdkPath | [QString][QString] | (Optional) The path to the folder containing **share/openbr/openbr.bib**. If no path is provided (default) OpenBR automatically searches: <ul> <li>The working directory</li> <li>The executable's location</li> </ul>
    useGui | bool | (Optional) Make OpenBR as a [QApplication][QApplication] instead of a [QCoreApplication][QCoreApplication]. Default is true.

* **output:** (void)
* **see:** [finalize](#context-static-finalize)
* **example:**

        int main(int argc, char \*argv[])
        {
            QApplication(argc, argv); // ONLY FOR QT PROJECTS
            br::Context::initialize(argc, argv);

            // ...

            br::Context::finalize();
            return 0;
        }

### void finalize() {: #context-static-finalize }

Call *once* at the end of the application to deallocate global variables.

* **function definition:**

        static void finalize();

* **parameters:** NONE
* **output:** (void)
* **see:** [initialize](#context-static-initialize)


### bool checkSDKPath(const [QString][QString] &sdkPath) {: #context-static-checksdkpath }

Check if a given SDK path is valid. A valid SDK satisfies

    exists(sdkPath + "share/openbr/openbr.bib")

* **function definition:**

        static bool checkSDKPath(const QString &sdkPath);

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    sdkPath | const [QString][QString] & | Possible sdk path to examine

* **output:** (bool) Returns true if the sdkPath + "share/openbr/openbr.bib" exists, otherwise returns false.
* **example:**

        // OpenBR is at /libs/openbr

        checkSDKPath("/libs/openbr/"); // returns true
        checkSDKPath("/libs/"); // returns false

### [QString][QString] about() {: #context-static-about }

Get a string with the name, version, and copyright of the project. This string is suitable for printing or terminal.

* **function definition:**

        static QString about();

* **parameters:** NONE
* **output:** ([QString][QString]) Returns a string containing the name, version and copyright of the project
* **example:**

        // Using OpenBR version 0.6.0
        Context::about(); // returns "OpenBR 0.6.0 Copyright (c) 2013 OpenBiometrics. All rights reserved."

### [QString][QString] version() {: #context-static-version }

Get the version of the SDK.

* **function definition:**

        static QString version();

* **parameters:** NONE
* **output:** ([QString][QString]) Returns a string containing the version of the OpenBR SDK. The string has the format *<MajorVersion\>*\.*<MinorVersion\>*\.*<PatchVersion\>*
* **example:**

        // Using OpenBR version 0.6.0
        Context::version(); // returns "0.6.0"

### [QString][QString] scratchPath() {: #context-static-scratchpath }

Get the scratch directory used by OpenBR. This directory should be used as the root directory for managing temporary files and providing process persistence.

* **function definition:**

        static QString scratchPath();

* **parameters:** NONE
* **output:** ([QString][QString]) Returns a string pointing to the OpenBR scratch directory. The string has the format *<path/to/user/home\><OpenBR-\><MajorVersion\>*\.*<MinorVersion\>*.
* **see:** [version](#context-static-version)
* **example:**

        // Using OpenBR version 0.6.0
        Context::scratchPath(); // returns "/path/to/user/home/OpenBR-0.6"

### [QStringList][QStringList] objects(const char \*abstractions = ".\*", const char \*implementations = ".\*", bool parameters = true) {: #context-static-objects }

Get a collection of objects in OpenBR that match provided regular expressions. This function uses [QRegExp][QRegExp] syntax.

* **function definition:**

        static QStringList objects(const char *abstractions = ".*", const char *implementations = ".*", bool parameters = true)

* **parameters:**

        Parameter | Type | Description
        --- | --- | ---
        abstractions | const char \* | (Optional) Regular expression of the abstractions to search. Default is ".\*"
        implementations | const char \* | (Optional) Regular expression of the implementations to search. Default is ".\*".
        parameters | bool | (Optional) If true include parameters after object name. Default is true.

* **output:** ([QStringList][QStringList]) Return names and parameters for the requested objects. Each object is newline separated. Arguments are separated from the object name with tabs.
* **example:**

        // Find all 'Rnd' Transforms
        Context::objects("Transform", "Rnd.*", false); // returns ["RndPoint", "RndRegion", "RndRotate", "RndSample", "RndSubspace"]

<!-- no italics* -->

---

## Functions {: #context-functions }


### bool contains(const [QString][QString] &name) {: #context-function-contains }

Check if a property exists in the [global metadata](#context).

* **function definition:**

        bool contains(const QString &name);

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    name | const [QString][QString] & | [Metadata](#context) key. It must be queryable using [QObject::property][QObject::property].

* **output:** (bool) Returns true if the provided key is a global property.
* **see:** [setProperty](#context-function-setproperty)
* **example:**

        Globals->contains("path"); // returns true
        Globals->contains("key"); // returns false


### void setProperty(const [QString][QString] &key, const [QString][QString] &value) {: #context-function-setproperty }

Set a global property.

* **function definition:**

        void setProperty(const QString &key, const QString &value)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | [Metadata](#context) key
    value | const [QString][QString] & | Value to be added to the [Metadata](#context)

* **output:** (void)
* **see:** [contains](#context-function-contains)
* **example:**

        Globals->contains("key"); // returns false
        Globals->setProperty("key", "value");
        Globals->contains("key"); // returns true


### void printStatus() {: #context-function-printstatus }

Prints the current progress statistics to **stdout**.

* **function definition:**

void printStatus();

* **parameters:** NONE
* **output:** (void)
* **see:** [progress](#context-function-progress)
* **example:**

        Globals->printStatus(); // returns 00.00%  ELAPSED=00:00:00  REMAINING=99:99:99  COUNT=0


### int timeRemaining() const {: #context-function-timeremaining }

Get the time remaining in seconds of a call to [Train](#function-train), [Enroll](#function-enroll-1) or [Compare](#function-compare).

* **function defintion:**

        int timeRemaining() const;

* **parameters:** NONE
* **output:** (int) Returns the estimated time remaining in the currently running process. If not process is running returns -1.

### float progress() {: #context-function-progress }

Get the completion percentage of a call to [Train](#function-train), [Enroll](#function-enroll-1), or [Compare](#function-compare).

* **function definition:**

        float progress() const;

* **parameters:** NONE
* **output:** (float) Returns the fraction of the currently running job that has been completed.

---

# Initializer

Inherits [Object](#object)

Plugin base class for initializing resources. On startup (the call to [Context](#context)::(initialize)(#context-static-initialize)), OpenBR will call [initialize](#initializer-function-initialize) on every Initializer that has been registered with the [Factory](#factory). On shutdown (the call to [Context](#context)::(finalize)(#context-static-finalize)), OpenBR will call [finalize](#initializer-function-finalize) on every registered initializer.

The general use case for initializers is to launch shared contexts for third party integrations into OpenBR. These cannot be launched during [Transform](#transform)::(init)(#transform-function-init) for example, because multiple instances of the [Transform](#transform) object could exist across multiple threads.

## Members {: #initializer-members }

NONE

## Constructors {: #initializer-constructors }

Destructor | Description
--- | ---
virtual ~Initializer() | Virtual function. Default destructor.

## Static Functions {: #initializer-static-functions }

NONE

## Functions {: #initializer-functions }

### virtual void initialize() const {: #initializer-function-initialize }

Initialize

* **function definition:**

        virtual void initialize() const = 0

---

# Transform

Inherits [Object](#object)

---

# UntrainableTransform

Inherits [Object](#object)

---

# MetaTransform

Inherits [Object](#object)

---

# UntrainableMetaTransform

Inherits [Object](#object)

---

# MetadataTransform

Inherits [Object](#object)

---

# UntrainableMetadataTransform

Inherits [Object](#object)

---

# TimeVaryingTransform

Inherits [Object](#object)

---

# Distance

Inherits [Object](#object)

---

# UntrainableDistance

Inherits [Object](#object)

---

# Output

Inherits from [Object](#object)

## Properties {: #output-properties }

Property | Type | Description
--- | --- | ---
<a class="table-anchor" id=output-properties-blockrows></a>blockRows | int | DOCUMENT ME
<a class="table-anchor" id=output-properties-blockcols></a>blockCols | int | DOCUMENT ME

## Members {: #output-members }

Member | Type | Description
--- | --- | ---
<a class="table-anchor" id=output-members-targetfiles></a>targetFiles | [FileList][#filelist] | DOCUMENT ME
<a class="table-anchor" id=output-members-queryfiles></a>queryFiles | [FileList](#filelist) | DOCUMENT ME
<a class="table-anchor" id=output-members-selfsimilar></a>selfSimilar | bool | DOCUMENT ME
<a class="table-anchor" id=output-members-next></a>next | [QSharedPointer][QSharedPointer]<[Output](#output)> | DOCUMENT ME
<a class="table-anchor" id=output-members-offset></a>offset | [QPoint][QPoint] | DOCUMENT ME

## Constructors {: #output-constructors }

Constructor \| Destructor | Description
--- | ---
virtual ~Output() | DOCUMENT ME

## Static Functions {: #output-static-functions }

### Output \*make(const [File][#file] &file, const [FileList](#filelist) &targetFiles, const [FileList](#filelist) &queryFiles) {: #output-function-make}

DOCUMENT ME

* **function definition:**

		static Output *make(const File &file, const FileList &targetFiles, const FileList &queryFiles)

* **parameters:**

	Parameter | Type | Description
	--- | --- | ---
	file | const [File](#file) & | DOCUMENT ME
	targetFiles | const [FileList](#filelist) & | DOCUMENT ME
	queryFiles | const [FileList](#filelist) & | DOCUMENT ME

* **output:** ([Output](#output)) DOCUMENT ME


## Functions {: #output-functions }

### virtual void initialize(const [FileList](#filelist) &targetFiles, const [FileList](#filelist) &queryFiles) {: #output-function-initialize }

DOCUMENT ME

* **function definition:**

		virtual void initialize(const [FileList](#filelist) &targetFiles, const [FileList](#filelist) &queryFiles)

* **parameters:**

	Parameter | Type | Description
	--- | --- | ---
	targetFiles | const [FileList](#filelist) & | DOCUMENT ME
	queryFiles | const [FileList](#filelist) & | DOCUMENT ME

* **output:** (void) DOCUMENT ME


### virtual void setBlock(int rowBlock, int columnBlock) {: #output-function-setblock }

DOCUMENT ME

* **function definition:**

		virtual void setBlock(int rowBlock, int columnBlock)

* **parameters:**

	Parameter | Type | Description
	--- | --- | ---
	rowBlock | int | DOCUMENT ME
	columnBlock | int | DOCUMENT ME

* **output:** (void) DOCUMENT ME


### virtual void setRelative(float value, int i, int j) {: #output-function-setrelative }

DOCUMENT ME

* **function definition:**

		virtual void setRelative(float value, int i, int j)

* **parameters:**

	Parameter | Type | Description
	--- | --- | ---
	value | float | DOCUMENT ME
	i | int | DOCUMENT ME
	j | int | DOCUMENT ME

* **output:** (void) DOCUMENT ME


### virtual void set(float value, int i, int j) = 0 {: #output-function-set }

DOCUMENT ME

* **function definition:**

		virtual void set(float value, int i, int j) = 0

* **parameters:**

	Parameter | Type | Description
	--- | --- | ---
	value | float | DOCUMENT ME
	i | int | DOCUMENT ME
	j | int | DOCUMENT ME

* **output:** (void) DOCUMENT ME

---

# MatrixOutput

Inherits [Object](#object)

---

# Format

Inherits [Object](#object)

---

# Gallery

Inherits [Object](#object)

---

# FileGallery

Inherits [Object](#object)

---

# Representation

Inherits [Object](#object).

---

# Classifier

Inherits [Object](#object)

[Qt]: http://qt-project.org/ "Qt"
[QApplication]: http://doc.qt.io/qt-5/qapplication.html "QApplication"
[QCoreApplication]: http://doc.qt.io/qt-5/qcoreapplication.html "QCoreApplication"

[QString]: http://doc.qt.io/qt-5/QString.html "QString"
[QStringList]: http://doc.qt.io/qt-5/qstringlist.html "QStringList"

[QList]: http://doc.qt.io/qt-5/QList.html "QList"
[QMap]: http://doc.qt.io/qt-5/qmap.html "QMap"
[QHash]: http://doc.qt.io/qt-5/qhash.html "QHash"

[QRectF]: http://doc.qt.io/qt-5/qrectf.html "QRectF"
[QPoint]: http://doc.qt.io/qt-5/qpoint.html "QPoint"
[QPointF]: http://doc.qt.io/qt-5/qpointf.html "QPointF"

[QVariant]: http://doc.qt.io/qt-5/qvariant.html "QVariant"
[QVariantList]: http://doc.qt.io/qt-5/qvariant.html#QVariantList-typedef "QVariantList"
[QVariantMap]: http://doc.qt.io/qt-5/qvariant.html#QVariantMap-typedef "QVariantMap"

[QRegExp]: http://doc.qt.io/qt-5/QRegExp.html "QRegExp"
[QThread]: http://doc.qt.io/qt-5/qthread.html "QThread"
[QFile]: http://doc.qt.io/qt-5/qfile.html "QFile"

[QSharedPointer]: http://doc.qt.io/qt-5/qsharedpointer.html "QSharedPointer"

[QTime]: http://doc.qt.io/qt-5/QTime.html "QTime"
[QDebug]: http://doc.qt.io/qt-5/qdebug.html "QDebug"
[QDataStream]: http://doc.qt.io/qt-5/qdatastream.html "QDataStream"
[QByteArray]: http://doc.qt.io/qt-5/qbytearray.html "QByteArray"

[Mat]: http://docs.opencv.org/modules/core/doc/basic_structures.html#mat "Mat"
[Rect]: http://docs.opencv.org/modules/core/doc/basic_structures.html#rect "Rect"
[InputArray]: http://docs.opencv.org/modules/core/doc/basic_structures.html#inputarray "InputArray"
[OutputArray]: http://docs.opencv.org/modules/core/doc/basic_structures.html#outputarray "OutputArray"
