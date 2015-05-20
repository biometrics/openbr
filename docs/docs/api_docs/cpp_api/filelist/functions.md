## [QStringList][QStringList] flat() const {: #flat }

Calls [flat](../file/functions.md#flat) on every [File](../file/file.md) in the list and returns the resulting strings as a [QStringList][QStringList].

* **function definition:**

        QStringList flat() const

* **parameters:** NONE
* **output:** ([QStringList][QStringList]) Returns a list of the output of calling [flat](../file/functions.md#flat) on each [File](../file/file.md)
* **example:**

        File f1("picture1.jpg"), f2("picture2.jpg");
        f1.set("Key", QString("Value"));

        FileList fList(QList<File>() << f1 << f2);
        fList.flat(); // returns ["picture1.jpg[Key=Value]", "picture2.jpg"]


## [QStringList][QStringList] names() const {: #names }

Get the [names](../file/members.md#name) of every [File](../file/file.md) in the list.

* **function definition:**

        QStringList names() const

* **parameters:** NONE
* **output:** ([QStringList][QStringList]) Returns the [name](../file/members.md#name) of every [File](../file/file.md) in the list
* **example:**

        File f1("picture1.jpg"), f2("picture2.jpg");
        f1.set("Key", QString("Value"));

        FileList fList(QList<File>() << f1 << f2);
        fList.names(); // returns ["picture1.jpg", "picture2.jpg"]


## void sort(const [QString][QString] &key) {: #sort }

Sort the [FileList](filelist.md) based on the values associated with a provided key in each [File](../file/file.md).

* **function definition:**

        void sort(const QString &key)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | Key to look up desired values in each [Files](../file/file.md) [metadata](../file/members.md#m_metadata)

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


## [QList][QList]&lt;int&gt; crossValidationPartitions() const {: #crossvalidationpartitions }

Get the cross-validation partion of each [File](../file/file.md) in the list. The partition is stored in each [File](../file/file.md) at [metadata](../file/members.md#m_metadata)["Partition"].

* **function definition:**

        QList<int> crossValidationPartitions() const

* **parameters:** NONE
* **output:** ([QList][QList]&lt;int&gt;) Returns the cross-validation partion of each [File](../file/file.md) as a list. If a [File](../file/file.md) does not have the "Partition" field in it's [metadata](../file/members.md#m_metadata) 0 is used.
* **example:**

        File f1, f2, f3;
        f1.set("Partition", QVariant::fromValue<int>(1));
        f3.set("Partition", QVariant::fromValue<int>(3));

        FileList fList(QList<File>() << f1 << f2 << f3);
        fList.crossValidationPartitions(); // returns [1, 0, 3]


## int failures() const {: #failures }

Get the number of [Files](../file/file.md) in the list that have [failed to enroll](../file/members.md#fte).

* **function definition:**

        int failures() const

* **parameters:** NONE
* **output:** (int) Returns the number of [Files](../file/file.md) that have [fte](../file/members.md#fte) equal true.
* **example:**

        File f1, f2, f3;
        f1.fte = false;
        f2.fte = true;
        f3.fte = true;

        FileList fList(QList<File>() << f1 << f2 << f3);
        fList.failures(); // returns 2

<!-- Links -->
[QList]: http://doc.qt.io/qt-5/QList.html "QList"
[QString]: http://doc.qt.io/qt-5/QString.html "QString"
[QStringList]: http://doc.qt.io/qt-5/qstringlist.html "QStringList"
