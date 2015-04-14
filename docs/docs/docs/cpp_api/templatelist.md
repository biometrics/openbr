<!-- TEMPLATELIST -->

Inherits [QList][QList]&lt;[Template](#template)&gt;.

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
