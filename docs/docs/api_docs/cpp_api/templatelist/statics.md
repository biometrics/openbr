## [TemplateList](templatelist.md) fromGallery(const [File](../file/file.md) &gallery) {: #fromgallery }

Create a [TemplateList](templatelist.md) from a gallery [File](../file/file.md).

* **function definition:**

        static TemplateList fromGallery(const File &gallery)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    gallery | const [File](../file/file.md) & | Gallery [file](../file/file.md) to be enrolled.

* **output:** ([TemplateList](templatelist.md)) Returns a [TemplateList](templatelist.md) created by enrolling the gallery.


## [TemplateList](templatelist.md) fromBuffer(const [QByteArray][QByteArray] &buffer) {: #frombuffer }

Create a template from a memory buffer of individual templates. This is compatible with **.gal** galleries.

* **function definition:**

        static TemplateList fromBuffer(const QByteArray &buffer)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    buffer | const [QByteArray][QByteArray] & | Raw data buffer to be enrolled

* **output:** ([TemplateList](templatelist.md)) Returns a [TemplateList](templatelist.md) created by enrolling the buffer


## [TemplateList](templatelist.md) relabel(const [TemplateList](templatelist.md) &tl, const [QString][QString] &propName, bool preserveIntegers) {: #relabel }

Relabel the values associated with a given key in the [metadata](../file/members.md#m_metadata) of a provided [TemplateList](templatelist.md). The values are relabeled to be between [0, numClasses-1]. If preserveIntegers is true and the [metadata](../file/members.md#m_metadata) can be converted to integers then numClasses equals the maximum value in the values. Otherwise, numClasses equals the number of unique values. The relabeled values are stored in the "Label" field of the returned [TemplateList](templatelist.md).

* **function definition:**

        static TemplateList relabel(const TemplateList &tl, const QString &propName, bool preserveIntegers)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    tl | const [TemplateList](templatelist.md) & | [TemplateList](templatelist.md) to be relabeled
    propName | const [QString][QString] & | [Metadata](../file/members.md#m_metadata) key
    preserveIntegers | bool | If true attempt to use the [metadata](../file/members.md#m_metadata) values as the relabeled values. Otherwise use the number of unique values.

* **output:** ([TemplateList](templatelist.md)) Returns a [TemplateList](templatelist.md) identical to the input [TemplateList](templatelist.md) but with the new labels appended to the [metadata](../file/members.md#m_metadata) using the "Label" key.
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

<!-- Links -->
[QByteArray]: http://doc.qt.io/qt-5/qbytearray.html "QByteArray"
[QString]: http://doc.qt.io/qt-5/QString.html "QString"
