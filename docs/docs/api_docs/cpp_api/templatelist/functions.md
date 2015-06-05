## [QList][QList]&lt;int&gt; indexProperty(const [QString][QString] &propName, [QHash][QHash]&lt;[QString][QString], int&gt; &valueMap, [QHash][QHash]&lt;int, [QVariant][QVariant]&gt; &reverseLookup) const {: #indexproperty-1 }

Convert [metadata](../file/members.md#m_metadata) values associated with **propName** to integers. Each unique value gets its own integer. This is useful in many classification problems where nominal data (e.g "Male", "Female") needs to represented with integers ("Male" = 0, "Female" = 1). **valueMap** and **reverseLookup** are created to allow easy conversion to the integer replacements and back.

* **function definition:**

        QList<int> indexProperty(const QString &propName, QHash<QString, int> &valueMap, QHash<int, QVariant> &reverseLookup) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    propName | const [QString][QString] & | [Metadata](../file/members.md#m_metadata) key
    valueMap | [QHash][QHash]&lt;[QString][QString], int&gt; & | A mapping from [metadata](../file/members.md#m_metadata) values to the equivalent unique index. [QStrings][QString] are used instead of [QVariant][QVariant] so comparison operators can be used. This is filled in by the function and can be provided empty.
    reverseLookup | [QHash][QHash]&lt;int, [QVariant][QVariant]&gt; & | A mapping from the unique index to the original value. This is the *reverse* mapping of the **valueMap**. This is filled in by the function and can be provided empty.

* **output:** ([QList][QList]&lt;int&gt;) Returns a list of unique integers that can be mapped to the [metadata](../file/members.md#m_metadata) values associated with **propName**. The integers can be mapped to their respective values using **valueMap** and the values can be mapped to the integers using **reverseLookup**.
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


## [QList][QList]&lt;int&gt; indexProperty(const [QString][QString] &propName, [QHash][QHash]&lt;[QString][QString], int&gt; \*valueMap=NULL, [QHash][QHash]&lt;int, [QVariant][QVariant]&gt; \*reverseLookup=NULL) const {: #indexproperty-2 }

Shortcut to call [indexProperty](#indexproperty-1) without **valueMap** or **reverseLookup** arguments.

* **function definition:**

        QList<int> indexProperty(const QString &propName, QHash<QString, int> * valueMap=NULL,QHash<int, QVariant> * reverseLookup = NULL) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    propName | const [QString][QString] & | [Metadata](../file/members.md#m_metadata) key
    valueMap | [QHash][QHash]&lt;[QString][QString], int&gt; \* | (Optional) A mapping from [metadata](../file/members.md#m_metadata) values to the equivalent unique index. [QStrings][QString] are used instead of [QVariant][QVariant] so comparison operators can be used. This is filled in by the function and can be provided empty.
    reverseLookup | [QHash][QHash]&lt;int, [QVariant][QVariant]&gt; \* | (Optional) A mapping from the unique index to the original value. This is the *reverse* mapping of the **valueMap**. This is filled in by the function and can be provided empty.

* **output:** ([QList][QList]&lt;int&gt;) Returns a list of unique integers that can be mapped to the [metadata](../file/members.md#m_metadata) values associated with **propName**. The integers can be mapped to their respective values using **valueMap** (if provided) and the values can be mapped to the integers using **reverseLookup** (if provided).


## [QList][QList]&lt;int&gt; applyIndex(const [QString][QString] &propName, const [QHash][QHash]&lt;[QString][QString], int&gt; &valueMap) const {: #applyindex }

Apply a mapping to convert non-integer values to integers. [Metadata](../file/members.md#m_metadata) values associated with **propName** are mapped through the given **valueMap**.

* **function definition:**

        QList<int> applyIndex(const QString &propName, const QHash<QString, int> &valueMap) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    propName | const [QString][QString] & | [Metadata](../file/members.md#m_metadata) key
    valueMap | const [QHash][QHash]&lt;[QString][QString], int&gt; & | (Optional) A mapping from [metadata](../file/members.md#m_metadata) values to the equivalent unique index. [QStrings][QString] are used instead of [QVariant][QVariant] so comparison operators can be used.

* **output:** ([Qlist][QList]&lt;int&gt;) Returns a list of integer values. The values are ordered in the same order as the [Templates](../template/template.md) in the list. The values are calculated like so:

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


## <tt>T</tt> bytes() const {: #bytes }

Get the total number of bytes in the [TemplateList](templatelist.md).

* **function definition:**

        template <typename T> T bytes() const

* **parameters:** NONE
* **output:** (<tt>T</tt>) Returns the sum of the bytes in each of the [Templates](../template/template.md) in the list. <tt>T</tt> is a user specified type. It is expected to be numeric (int, float etc.)
* **see:** [bytes](../template/functions.md#bytes)
* **example:**

        Template t1, t2;

        t1.append(Mat::ones(1, 1, CV_8U)); // 1 byte
        t1.append(Mat::ones(2, 2, CV_8U)); // 4 bytes
        t2.append(Mat::ones(3, 3, CV_8U)); // 9 bytes
        t2.append(Mat::ones(4, 4, CV_8U)); // 16 bytes

        TemplateList tList(QList<Template>() << t1 << t2);
        tList.bytes(); // returns 30


## [QList][QList]&lt;[Mat][Mat]&gt; data(int index = 0) const {: #data }

Get a list of matrices compiled from each [Template](../template/template.md) in the list.

* **function definition:**

        QList<cv::Mat> data(int index = 0) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    index | int | (Optional) Index into each [Template](../template/template.md) to select a [Mat][Mat]. Default is 0.

* **output:** ([QList][QList]&lt;[Mat][Mat]&gt;) Returns a list of [Mats][Mat]. One [Mat][Mat] is supplied by each [Template](../template/template.md) in the image at the specified index.
* **example:**

        Template t1, t2;

        t1.append(Mat::ones(1, 1, CV_8U));
        t1.append(Mat::zeros(1, 1, CV_8U));
        t2.append(Mat::ones(1, 1, CV_8U));
        t2.append(Mat::zeros(1, 1, CV_8U));

        TemplateList tList(QList<Template>() << t1 << t2);
        tList.data(); // returns ["1", "1"];
        tList.data(1); // returns ["0", "0"];


## [QList][QList]&lt;[TemplateList](templatelist.md)&gt; partition(const [QList][QList]&lt;int&gt; &partitionSizes) const {: #partition }

Divide the [TemplateList](templatelist.md) into a list of [TemplateLists](templatelist.md) partitions.

 * **function defintion:**

        QList<TemplateList> partition(const QList<int> &partitionSizes) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    partitionSizes | [QList][QList]&lt;int&gt; | A list of sizes for the partitions. The total number of partitions is equal to the length of this list. Each value in this list specifies the number of [Mats][Mat] that should be in each template of the associated partition. The sum of values in this list *must* equal the number of [Mats][Mat] in each [Template](../template/template.md) in the original [TemplateList](templatelist.md).

* **output:** ([QList][QList]&lt;[TemplateList](templatelist.md)&gt;) Returns a [QList][QList] of [TemplateLists](templatelist.md) of partitions. Each partition has length equal to the number of templates in the original [TemplateList](templatelist.md). Each [Template](../template/template.md) has length equal to the size specified in the associated value in **partitionSizes**.
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

## [FileList](../filelist/filelist.md) files() const {: #files }

Get a list of all the [Files](../file/file.md) in the [TemplateList](templatelist.md)

* **function definition:**

        FileList files() const

* **parameters:** NONE
* **output:** ([FileList](../filelist/filelist.md)) Returns a [FileList](../filelist/filelist.md) with the [file](../template/members.md#file) of each [Template](../template/template.md) in the [TemplateList](templatelist.md).
* **example:**

        Template t1("picture1.jpg"), t2("picture2.jpg");

        t1.file.set("Key", QVariant::fromValue<float>(1));
        t2.file.set("Key", QVariant::fromValue<float>(2));

        TemplateList tList(QList<Template>() << t1 << t2);

        tList.files(); // returns ["picture1.jpg[Key=1]", "picture2.jpg[Key=2]"]


## [FileList](../filelist/filelist.md) operator()() {: #operator-pp }

Shortcut call to [files](#files)

* **function definition:**

        FileList operator()() const

* **parameters:** NONE
* **output:** ([FileList](../filelist/filelist.md)) Returns a [FileList](../filelist/filelist.md) with the [file](../template/members.md#file) of each [Template](../template/template.md) in the [TemplateList](templatelist.md).
* **example:**

        Template t1("picture1.jpg"), t2("picture2.jpg");

        t1.file.set("Key", QVariant::fromValue<float>(1));
        t2.file.set("Key", QVariant::fromValue<float>(2));

        TemplateList tList(QList<Template>() << t1 << t2);

        tList.files(); // returns ["picture1.jpg[Key=1]", "picture2.jpg[Key=2]"]


## [QMap][QMap]&lt;T, int&gt; countValues(const [QString][QString] &propName, bool excludeFailures = false) const {: #countvalues }

Get the frequency of each unique value associated with a provided [metadata](../file/members.md#m_metadata) key.

* **function definition:**

template<typename T> QMap<T,int> countValues(const QString &propName, bool excludeFailures = false) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    propName | const [QString][QString] & | [Metadata](../file/members.md#m_metadata) key
    excludeFailures | bool | (Optional) Exclude [File](../file/file.md) [metadata](../file/members.md#m_metadata) if the [File](../file/file.md) has [fte](../file/members.md#fte) equal to true. Default is false

* **output:** ([QMap][QMap]&lt;<T, int&gt;) Returns a mapping between unique [metadata](../file/members.md#m_metadata) and their frequency.
* **example:**

        Template t1, t2, t3, t4;

        t1.file.set("Key", QString("Class 1"));
        t2.file.set("Key", QString("Class 2"));
        t3.file.set("Key", QString("Class 3"));
        t4.file.set("Key", QString("Class 1"));

        TemplateList tList(QList<Template>() << t1 << t2 << t3 << t4);

        tList.countValues<QString>("Key"); // returns QMap(("Class 1", 2), ("Class 2", 1), ("Class 3", 1))

## [TemplateList](templatelist.md) reduced() const {: #reduced }

Reduce the [Templates](../template/template.md) in the [TemplateList](templatelist.md) by merging them together.

* **function definition:**

        TemplateList reduced() const

* **parameters:** NONE
* **output:** ([TemplateList](templatelist.md)) Returns a [TemplateList](templatelist.md) with a single [Template](../template/template.md). The [Template](../template/template.md) is the result of calling [merge](../template/functions.md#merge) on every [Template](../template/template.md).
* **see:** [merge](../template/functions.md#merge)
* **example:**

        Template t1("picture1.jpg"), t2("picture2.jpg");

        t1.file.set("Key1", QString("Value1"));
        t2.file.set("Key2", QString("Value2"));

        TemplateList tList(QList<Template>() << t1 << t2);

        TemplateList reduced = tList.reduced();
        reduced.size(); // returns 1
        reduced.files(); // returns ["picture1.jpg;picture2.jpg[Key1=Value1, Key2=Value2, separator=;]"]

## [QList][QList]&lt;int&gt; find(const [QString][QString] &key, const T &value) {: #find }

Get the indices of every [Template](../template/template.md) that has a provided key value pairing in its [metadata](../file/members.md#m_metadata)

* **function definition:**

        template<typename T> QList<int> find(const QString &key, const <tt>T</tt> &value)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | [Metadata](../file/members.md#m_metadata) key to search for
    value | const <tt>T</tt> & | Value to search for. Both the **key** and value must match. <tt>T</tt> is a user specified type.

* **output:** ([QList][QList]&lt;int&gt;) Returns a list of indices for [Templates](../template/template.md) that contained the key-value pairing in their [metadata](../file/members.md#m_metadata)
* **example:**

        Template t1, t2, t3;

        t1.file.set("Key", QString("Value1"));
        t2.file.set("Key", QString("Value2"));
        t3.file.set("Key", QString("Value2"));

        TemplateList tList(QList<Template>() << t1 << t2 << t3);
        tList.find<QString>("Key", "Value2"); // returns [1, 2]

<!-- Links -->
[QList]: http://doc.qt.io/qt-5/QList.html "QList"
[QHash]: http://doc.qt.io/qt-5/qhash.html "QHash"
[QMap]: http://doc.qt.io/qt-5/qmap.html "QMap"
[QString]: http://doc.qt.io/qt-5/QString.html "QString"
[QVariant]: http://doc.qt.io/qt-5/qvariant.html "QVariant"
[Mat]: http://docs.opencv.org/modules/core/doc/basic_structures.html#mat "Mat"
