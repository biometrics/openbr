<!-- FILELIST -->

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
