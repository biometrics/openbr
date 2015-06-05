## operator [QString][QString]() {: #operator-qstring }

Convenience function that allows [Files](file.md) to be used as [QStrings][QString]

* **function definition:**

        inline operator QString() const

* **parameters:** NONE
* **output:** ([QString][QString]) returns [name](members.md#name).

## [QString][QString] flat() {: #flat }

Function to output files in string formats.

* **function definition:**

        QString flat() const

* **parameters:** NONE
* **output:** ([QString][QString]) returns the [file name](members.md#name) and [metadata](members.md#m_metadata) as a formated string. The format is *filename*[*key1=value1,key2=value2,...keyN=valueN*].
* **example:**

        File file("picture.jpg");
        file.set("Key1", QVariant::fromValue<float>(1));
        file.set("Key2", QVariant::fromValue<float>(2));

        file.flat(); // returns "picture.jpg[Key1=1,Key2=2]"


## [QString][QString] hash() {: #hash }

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


## [QStringList][QStringList] localKeys() {: #localkeys }

Function to get the private [metadata](members.md#m_metadata) keys.

* **function definition:**

        inline QStringList localKeys() const

* **parameters:** NONE
* **output:** ([QStringList][QStringList]) Returns a list of the local [metadata](members.md#m_metadata) keys. They are called local because they do not include the keys in the [global metadata](../context/context.md).
* **example:**

    File file("../path/to/pictures/picture.jpg");
    file.set("Key1", QVariant::fromValue<float>(1));
    file.set("Key2", QVariant::fromValue<float>(2));

    file.localKeys(); // returns [Key1, Key2]


## [QVariantMap][QVariantMap] localMetadata() {: #localmetadata }

Function to get the private [metadata](members.md#m_metadata).

* **function definition:**

        inline QVariantMap localMetadata() const

* **parameters:** NONE
* **output:** ([QVariantMap][QVariantMap]) Returns the local [metadata](members.md#m_metadata).
* **example:**

        File file("../path/to/pictures/picture.jpg");
        file.set("Key1", QVariant::fromValue<float>(1));
        file.set("Key2", QVariant::fromValue<float>(2));

        file.localMetadata(); // return QMap(("Key1", QVariant(float, 1)) ("Key2", QVariant(float, 2)))


## void append(const [QVariantMap][QVariantMap] &localMetadata) {: #append-1 }

Add new metadata fields to [metadata](members.md#m_metadata).

* **function definition:**

        void append(const QVariantMap &localMetadata)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    localMetadata | const [QVariantMap][QVariantMap] & | metadata to append to the local [metadata](members.md#m_metadata)

* **output:** (void)
* **example:**

        File f();
        f.set("Key1", QVariant::fromValue<float>(1));

        QVariantMap map;
        map.insert("Key2", QVariant::fromValue<float>(2));
        map.insert("Key3", QVariant::fromValue<float>(3));

        f.append(map);
        f.flat(); // returns "[Key1=1, Key2=2, Key3=3]"


## void append(const [File](file.md) &other) {: #append-2 }

Append another file using the **;** separator. The [File](file.md) [names](members.md#name) are combined with the separator in between them. The [metadata](members.md#m_metadata) fields are combined. An additional field describing the separator is appended to the [metadata](members.md#m_metadata).

* **function definition:**

        void append(const File &other)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    other | const [File](file.md) & | File to append

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


## [File](file.md) &operator+=(const [QMap][QMap]&lt;[QString][QString], [QVariant][QVariant]&gt; &other) {: #operator-pe-1 }

Shortcut operator to call [append](#append-1).

* **function definition:**

        inline File &operator+=(const QMap<QString, QVariant> &other)

* **parameters:**

    Parameter | Type | Description
    other | const [QMap][QMap]&lt;[QString][QString], [QVariant][QVariant]&gt; & | Metadata map to append to the local [metadata](members.md#m_metadata)

* **output:** ([File](file.md) &) Returns a reference to this file after the append occurs.
* **example:**

        File f();
        f.set("Key1", QVariant::fromValue<float>(1));

        QMap<QString, QVariant> map;
        map.insert("Key2", QVariant::fromValue<float>(2));
        map.insert("Key3", QVariant::fromValue<float>(3));

        f += map;
        f.flat(); // returns "[Key1=1, Key2=2, Key3=3]"


## [File](file.md) &operator+=(const [File](file.md) &other) {: #operator-pe-2 }

Shortcut operator to call [append](#append-2).

* **function definition:**

        inline File &operator+=(const File &other)

* **parameters:**

    Parameter | Type | Description
    other | const [File](file.md) & | File to append

* **output:** ([File](file.md) &) Returns a reference to this file after the append occurs.
* **example:**

        File f1("../path/to/pictures/picture1.jpg");
        f1.set("Key1", QVariant::fromValue<float>(1));

        File f2("../path/to/pictures/picture2.jpg");
        f2.set("Key2", QVariant::fromValue<float>(2));
        f2.set("Key3", QVariant::fromValue<float>(3));

        f1 += f2;
        f1.name; // return "../path/to/pictures/picture1.jpg;../path/to/pictures/picture2.jpg"
        f1.localKeys(); // returns "[Key1, Key2, Key3, separator]"


## [QList][QList]&lt;[File](file.md)&gt; split() {: #split-1 }

This function splits the [File](file.md) into multiple files and returns them as a list. This is done by parsing the file [name](members.md#name) and splitting on the separator located at [metadata](members.md#m_metadata)["separator"]. If "separator" is not a [metadata](members.md#m_metadata) key, the returned list has the original file as the only entry. Each new file has the same [metadata](members.md#m_metadata) as the original, pre-split, file.

* **function definition:**

        QList<File> split() const

* **parameters:** None
* **output:** ([QList][QList]&lt;[File](file.md)&gt;) List of split files
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


## [QList][QList]&lt;[File](file.md)&gt; split(const [QString][QString] &separator) {: #split-2 }

This function splits the file into multiple files and returns them as a list. This is done by parsing the file [name](members.md#name) and splitting on the given separator. Each new file has the same [metadata](members.md#m_metadata) as the original, pre-split, file.

* **function definition:**

        QList<File> split(const QString &separator) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    separator | const [QString][QString] & | Separator to split the file name on

* **output:** ([QList][QList]&lt;[File](file.md)&gt;) List of split files
* **example:**

        File f("../path/to/pictures/picture1.jpg,../path/to/pictures/picture2.jpg");
        f.set("Key1", QVariant::fromValue<float>(1));
        f.set("Key2", QVariant::fromValue<float>(2));

        f.split(","); // returns [../path/to/pictures/picture1.jpg[Key1=1, Key2=2],
                                  ../path/to/pictures/picture2.jpg[Key1=1, Key2=2]]


## void setParameter(int index, const [QVariant][QVariant] &value) {: #setparameter }

Insert a keyless value into the [metadata](members.md#m_metadata). Generic key of "ArgN" is used, where N is given as a parameter.

* **function definition:**

        inline void setParameter(int index, const QVariant &value)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    index | int | Number to append to generic key
    value | const [QVariant][QVariant] & | Value to add to the metadata

* **output:** (void)
* **see:** [containsParameter](#containsparameter), [getParameter](#getparameter)
* **example:**

        File f;
        f.set("Key1", QVariant::fromValue<float>(1));
        f.set("Key2", QVariant::fromValue<float>(2));

        f.setParameter(1, QVariant::fromValue<float>(3));
        f.setParameter(5, QVariant::fromValue<float>(4));

        f.flat(); // returns "[Key1=1, Key2=2, Arg1=3, Arg5=4]"


## bool containsParameter(int index) {: #containsparameter }

Check if the local [metadata](members.md#m_metadata) contains a keyless value.

* **function definition:**

        inline bool containsParameter(int index) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    index | int | Index of the keyless value to check for

* **output:** (bool) Returns true if the local [metadata](members.md#m_metadata) contains the keyless value, otherwise reutrns false.
* **see:** [setParameter](#setparameter), [getParameter](#getparameter)
* **example:**

        File f;
        f.setParameter(1, QVariant::fromValue<float>(1));
        f.setParameter(2, QVariant::fromValue<float>(2));

        f.containsParameter(1); // returns true
        f.containsParameter(2); // returns true
        f.containsParameter(3); // returns false


## [QVariant][QVariant] getParameter(int index) {: #getparameter }

Get a keyless value from the local [metadata](members.md#m_metadata). If the value does not exist an error is thrown.

* **function definition:**

        inline QVariant getParameter(int index) const

* **parameter:**

    Parameter | Type | Description
    --- | --- | ---
    index | int | Index of the keyless value to look up. If the index does not exist an error is thrown.

* **output:** ([QVariant][QVariant]) Returns the keyless value associated with the given index
* **see:** [setParameter](#setparameter), [containsParameter](#containsparameter)
* **example:**

        File f;
        f.setParameter(1, QVariant::fromValue<float>(1));
        f.setParameter(2, QVariant::fromValue<float>(2));

        f.getParameter(1); // returns 1
        f.getParameter(2); // returns 2
        f.getParameter(3); // error: index does not exist


## bool operator==(const char \*other) {: #operator-ee-1 }

Compare [name](members.md#name) against a c-style string.

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


## bool operator==(const [File](file.md) &other) {: #operator-ee-2 }

Compare [name](members.md#name) and [metadata](members.md#m_metadata) against another file name and metadata.

* **function definition:**

        inline bool operator==(const File &other) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    other | const [File](file.md) & | File to compare against

* **output:** (bool) Returns true if the names and metadata are equal, false otherwise.
* **example:**

        File f1("picture1.jpg");
        File f2("picture1.jpg");

        f1 == f2; // returns true

        f1.set("Key1", QVariant::fromValue<float>(1));
        f2.set("Key2", QVariant::fromValue<float>(2));

        f1 == f2; // returns false (metadata doesn't match)


## bool operator!=(const [File](file.md) &other) {: #operator-ne }

Compare [name](members.md#name) and [metadata](members.md#m_metadata) against another file name and metadata.

* **function definition:**

        inline bool operator!=(const File &other) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    other | const [File](file.md) & | File to compare against

* **output:** (bool) Returns true if the names and metadata are not equal, false otherwise.
* **example:**

        File f1("picture1.jpg");
        File f2("picture1.jpg");

        f1 != f2; // returns false

        f1.set("Key1", QVariant::fromValue<float>(1));
        f2.set("Key2", QVariant::fromValue<float>(2));

        f1 != f2; // returns true (metadata doesn't match)


## bool operator<(const [File](file.md) &other) {: #operator-lt }

Compare [name](members.md#name) against another file name.

* **function definition:**

        inline bool operator<(const File &other) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    other | const [File](file.md) & | File to compare against

* **output:** (bool) Returns true if [name](members.md#name) < others.name


## bool operator<=(const [File](file.md) &other) {: #operator-lte }

Compare [name](members.md#name) against another file name.

* **function definition:**

        inline bool operator<=(const File &other) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    other | const [File](file.md) & | File to compare against

* **output:** (bool) Returns true if [name](members.md#name) <= others.name


## bool operator>(const [File](file.md) &other) {: #operator-gt }

Compare [name](members.md#name) against another file name.

* **function definition:**

        inline bool operator>(const File &other) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    other | const [File](file.md) & | File to compare against

* **output:** (bool) Returns true if [name](members.md#name) > others.name


## bool operator>=(const [File](file.md) &other) {: #operator-gte }

Compare [name](members.md#name) against another file name.

* **function definition:**

        inline bool operator>=(const File &other) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    other | const [File](file.md) & | File to compare against

* **output:** (bool) Returns true if [name](members.md#name) >= others.name


## bool isNull() {: #isnull }

Check if the file is null.

* **function definition:**

        inline bool isNull() const

* **parameters:** NONE
* **output:** (bool) Returns true if [name](members.md#name) and [metadata](members.md#m_metadata) are empty, false otherwise.
* **example:**

        File f;
        f.isNull(); // returns true

        f.set("Key1", QVariant::fromValue<float>(1));
        f.isNull(); // returns false


## bool isTerminal() {: #isterminal }

Checks if the value of [name](members.md#name) == "terminal".

* **function definition:**

        inline bool isTerminal() const

* **parameters:** NONE
* **output:** (bool) Returns true if [name](members.md#name) == "terminal", false otherwise.
* **example:**

        File f1("terminal"), f2("not_terminal");

        f1.isTerminal(); // returns true
        f2.isTerminal(); // returns false


## bool exists() {: #exists }

Check if the file exists on disk.

* **function definition:**

        inline bool exists() const

* **parameters:** NONE
* **output:** Returns true if [name](members.md#name) exists on disk, false otherwise.
* **example:**

    File f1("/path/to/file/that/exists"), f2("/path/to/non/existant/file");

    f1.exists(); // returns true
    f2.exists(); // returns false


## [QString][QString] fileName() {: #filename }

Get the file's base name and extension.

* **function definition:**

        inline QString fileName() const

* **parameters:** NONE
* **output:** ([QString][QString]) Returns the base name + extension of [name](members.md#name)
* **example:**

        File file("../path/to/pictures/picture.jpg");
        file.fileName(); // returns "picture.jpg"


## [QString][QString] baseName() {: #basename }

Get the file's base name.

* **function definition:**

        inline QString baseName() const

* **parameters:** NONE
* **output:** ([QString][QString]) Returns the base name of [name](members.md#name)
* **example:**

        File file("../path/to/pictures/picture.jpg");
        file.baseName(); // returns "picture"


## [QString][QString] suffix() {: #suffix }

Get the file's extension.

* **function definition:**

        inline QString suffix() const

* **parameters:** NONE
* **output:** ([QString][QString]) Returns the extension of [name](members.md#name)
* **example:**

        File file("../path/to/pictures/picture.jpg");
        file.suffix(); // returns "jpg"


## [QString][QString] path() {: #path }

Get the path of the file without the name.

* **function definition:**

        inline QString path() const

* **parameters:** NONE
* **output:** ([QString][QString]) Returns the path of [name](members.md#name).
* **example:**

        File file("../path/to/pictures/picture.jpg");
        file.suffix(); // returns "../path/to/pictures"


## [QString][QString] resolved() {: #resolved }

Get the full path for the file. This is done in three steps:

1. If [name](members.md#name) exists, return [name](members.md#name).
2. Prepend each path stored in [Globals->path](../context/members.md#path) to [name](members.md#name). If the combined name exists then it is returned.
3. Prepend each path stored in [Globals->path](../context/members.md#path) to [fileName](#filename). If the combined name exists then it is returned.

If none of the attempted names exist, [name](members.md#name) is returned unmodified.

* **function definition:**

        QString resolved() const

* **parameters:** NONE
* **output:** ([QString][QString]) Returns the resolved string if it can be created. Otherwise it returns [name](members.md#name)


## bool contains(const [QString][QString] &key) {: #contains-1 }

Check if a given key is in the local [metadata](members.md#m_metadata).

* **function definition:**

        bool contains(const QString &key) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | Key to check the [metadata](members.md#m_metadata) for

* **output:** (bool) Returns true if the given key is in the [metadata](members.md#m_metadata), false otherwise.
* **example:**

        File file;
        file.set("Key1", QVariant::fromValue<float>(1));

        file.contains("Key1"); // returns true
        file.contains("Key2"); // returns false


## bool contains(const [QStringList][QStringList] &keys) {: #contains-2 }

Check if a list of keys is in the local [metadata](members.md#m_metadata).

* **function definition:**

        bool contains(const QStringList &keys) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    keys | const [QStringList][QStringList] & | Keys to check the [metadata](members.md#m_metadata) for

* **output:** (bool) Returns true if *all* of the given keys are in the [metadata](members.md#m_metadata), false otherwise.
* **example:**

        File file;
        file.set("Key1", QVariant::fromValue<float>(1));
        file.set("Key2", QVariant::fromValue<float>(2));

        file.contains(QStringList("Key1")); // returns true
        file.contains(QStringList() << "Key1" << "Key2") // returns true
        file.contains(QStringList() << "Key1" << "Key3"); // returns false


## [QVariant][QVariant] value(const [QString][QString] &key) {: #value }

Get the value associated with a given key from the [metadata](members.md#m_metadata). If the key is not found in the [local metadata](members.md#m_metadata), the [global metadata](../context/context.md) is searched. In a special case, the key can be "name". This returns the file's [name](members.md#name).

* **function description:**

        QVariant value(const QString &key) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | Key to look up the value in the [local metadata](members.md#m_metadata) or [global metadata](../context/context.md). The key can also be "name".

* **output:** ([QVariant][QVariant]) Returns the key associated with the value from either the [local](members.md#m_metadata) or [global](../context/context.md) metadata. If the key is "name", [name](members.md#name) is returned.
* **example:**

        File file;
        file.set("Key1", QVariant::fromValue<float>(1));
        file.value("Key1"); // returns QVariant(float, 1)


## void set(const [QString][QString] &key, const [QVariant][QVariant] &value) {: #set-1 }

Insert a value into the [metadata](members.md#m_metadata) using a provided key. If the key already exists the new value will override the old one.

* **function description:**

        inline void set(const QString &key, const QVariant &value)

* **parameters:**

    Parameters | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | Key to store the given value in the [metadata](members.md#m_metadata)
    value | const [QVariant][QVariant] & | Value to be stored

* **output:** (void)
* **example:**

        File f;
        f.flat(); // returns ""

        f.set("Key1", QVariant::fromValue<float>(1));
        f.flat(); // returns "[Key1=1]"


## void set(const [QString][QString] &key, const [QString][QString] &value) {: #set-2 }

Insert a value into the [metadata](members.md#m_metadata) using a provided key. If the key already exists the new value will override the old one.

* **function description:**

        void set(const QString &key, const QString &value)

* **parameters:**

    Parameters | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | Key to store the given value in the [metadata](members.md#m_metadata)
    value | const [QString][QString] & | Value to be stored

* **output:** (void)
* **example:**

        File f;
        f.flat(); // returns ""

        f.set("Key1", QString("1"));
        f.flat(); // returns "[Key1=1]"


## void setList(const [QString][QString] &key, const [QList][QList]&lt;T&gt; &value) {: #setlist }

Insert a list into the [metadata](members.md#m_metadata) using a provided key. If the key already exists the new value will override the old one. The value should be queried with [getList](#getlist-1) instead of [get](#get-1).

* **function description:**

        template <typename T> void setList(const QString &key, const QList<T> &value)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | Key to store the given value in the [metadata](members.md#m_metadata)
    value | const [QList][QList]&lt;T&gt; | List to be stored

* **output:** (void)
* **see:** [getList](#getlist-1), [get](#get-1)
* **example:**

        File file;

        QList<float> list = QList<float>() << 1 << 2 << 3;
        file.setList<float>("List", list);
        file.getList<float>("List"); // return [1., 2. 3.]


## void remove(const [QString][QString] &key) {: #remove }

Remove a key-value pair from the [metadata](members.md#m_metadata)

* **function description:**

        inline void remove(const QString &key)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | Key to be removed from [metadata](members.md#m_metadata) along with its associated value.

* **output:** (void)
* **example:**

        File f;
        f.set("Key1", QVariant::fromValue<float>(1));
        f.set("Key2", QVariant::fromValue<float>(2));

        f.flat(); // returns "[Key1=1, Key2=2]"

        f.remove("Key1");
        f.flat(); // returns "[Key2=2]"


## T get(const [QString][QString] &key) {: #get-1 }

Get a value from the [metadata](members.md#m_metadata) using a provided key. If the key does not exist or the value cannot be converted to a user specified type an error is thrown.

* **function definition:**

        template <typename T> T get(const QString &key) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | Key to retrieve a value from [metadata](members.md#m_metadata)

* **output:** (<tt>T</tt>) Returns a value of type <tt>T</tt>. <tt>T</tt> is a user specified type. The value associated with the given key must be convertable to <tt>T</tt>.
* **see:** [get](#get-2), [getList](#getlist-1)
* **example:**

        File f;
        f.set("Key1", QVariant::fromValue<float>(1));

        f.get<float>("Key1");  // returns 1
        f.get<float>("Key2");  // Error: Key2 is not in the metadata
        f.get<QRectF>("Key1"); // Error: A float can't be converted to a QRectF

## T get(const [QString][QString] &key, const T &defaultValue) {: #get-2 }

Get a value from the [metadata](members.md#m_metadata) using a provided key. If the key does not exist or the value cannot be converted to user specified type a provided default value is returned instead.

* **function definition:**

        template <typename T> T get(const QString &key, const T &defaultValue)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | Key to retrieve a value from the [metadata](members.md#m_metadata)
    defaultValue | const T & | Default value to be returned if the key does not exist or found value cannot be converted to <tt>T</tt>. <tt>T</tt> is a user specified type.

* **output:** (<tt>T</tt>) Returns a value of type <tt>T</tt>. <tt>T</tt> is a user specified type. If the value associated with the key is invalid, the provided default value is returned instead.
* **see:** [get](#get-1), [getList](#getlist-1)
* **example:**

        File f;
        f.set("Key1", QVariant::fromValue<float>(1));

        f.get<float>("Key1", 5);  // returns 1
        f.get<float>("Key2", 5);  // returns 5
        f.get<QRectF>("Key1", QRectF(0, 0, 10, 10)); // returns QRectF(0, 0, 10x10)


## bool getBool(const [QString][QString] &key, bool defaultValue = false) {: #getbool }

Get a boolean value from the [metadata](members.md#m_metadata) using a provided key. If the key is not in the [metadata](members.md#m_metadata) a provided default value is returned. If the key is in the metadata but the value cannot be converted to a bool true is returned. If the key is found and the value can be converted to a bool the value is returned.

* **function definition:**

        bool getBool(const QString &key, bool defaultValue = false) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | Key to retrieve a value from the [metadata](members.md#m_metadata)
    defaultValue | bool | (Optional) Default value to be returned if the key is not in the [metadata](members.md#m_metadata).

* **output:** (bool) If the key *is not* in the [metadata](members.md#m_metadata) the provided default value is returned. If the key *is* in the [metadata](members.md#m_metadata) but the associated value *cannot* be converted to a bool true is returned. If the key *is* in the [metadata](members.md#m_metadata) and the associated value *can* be converted to a bool, that value is returned.
* **see:** [get](#get-2)
* **example:**

        File f;
        f.set("Key1", QVariant::fromValue<bool>(true));
        f.set("Key2", QVariant::fromValue<float>(10));

        f.getBool("Key1");       // returns true
        f.getBool("Key2")        // returns true (key found)
        f.getBool("Key3");       // returns false (default value)
        f.getBool("Key3", true); // returns true (default value)


## [QList][QList]&lt;T&gt; getList(const [QString][QString] &key) {: #getlist-1 }

Get a list from the [metadata](members.md#m_metadata) using a provided key. If the key does not exist or the elements of the list cannot be converted to a user specified type an error is thrown.

* **function definition:**

        template <typename T> QList<T> getList(const QString &key) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | Key to retrieve a value from the [metadata](members.md#m_metadata)

* **output:** ([QList][QList]&lt;<tt>T</tt>&gt;) Returns a list of values of a user specified type.
* **see:** [setList](#setlist), [get](#get-1)
* **example:**

        File file;

        QList<float> list = QList<float>() << 1 << 2 << 3;
        file.setList<float>("List", list);

        file.getList<float>("List");  // return [1., 2. 3.]
        file.getList<QRectF>("List"); // Error: float cannot be converted to QRectF
        file.getList<float>("Key");   // Error: key doesn't exist


## [QList][QList]&lt;T&gt; getList(const [QString][QString] &key, const [QList][QList]&lt;T&gt; defaultValue) {: #getlist-2 }

Get a list from the [metadata](members.md#m_metadata) using a provided key. If the key does not exist or the elements of the list cannot be converted to a user specified type a provided default value is returned.

* **function definition:**

template <typename T> QList<T> getList(const QString &key, const QList<T> defaultValue) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    key | const [QString][QString] & | Key to retrieve a value from the [metadata](members.md#m_metadata)
    defaultValue | [QList][QList]&lt;<tt>T</tt> | (Optional) Default value to be returned if the key is not in the [metadata](members.md#m_metadata).

* **output:** ([QList][QList]&lt;<tt>T</tt>&gt;) Returns a list of values of user specified type. If key is not in the [metadata](members.md#m_metadata) or if the value cannot be converted to a [QList][QList]&lt;<tt>T</tt>&gt; the default value is returned.
* **see:** [getList](#getlist-1)
* **example:**

        File file;

        QList<float> list = QList<float>() << 1 << 2 << 3;
        file.setList<float>("List", list);

        file.getList<float>("List", QList<float>());                  // return [1., 2. 3.]
        file.getList<QRectF>("List", QList<QRectF>());                // return []
        file.getList<float>("Key", QList<float>() << 1 << 2 << 3);    // return [1., 2., 3.]


## [QList][QList]&lt;[QPointF][QPointF]&gt; namedPoints() {: #namedpoints }

Find values in the [metadata](members.md#m_metadata) that can be converted into [QPointF][QPointF]'s. Values stored as [QList][QList]&lt;[QPointF][QPointF]&gt; *will not** be returned.

* **function definition:**

        QList<QPointF> namedPoints() const

* **parameters:** NONE
* **output:** ([QList][QList]&lt;[QPointF][QPointF]&gt;) Returns a list of points that can be converted from [metadata](members.md#m_metadata) values.
* **example:**

        File file;
        file.set("Key1", QVariant::fromValue<QPointF>(QPointF(1, 1)));
        file.set("Key2", QVariant::fromValue<QPointF>(QPointF(2, 2)));
        file.set("Points", QVariant::fromValue<QPointF>(QPointF(3, 3)))

        f.namedPoints(); // returns [QPointF(1, 1), QPointF(2, 2), QPointF(3, 3)]

        file.setPoints(QList<QPointF>() << QPointF(3, 3)); // changes metadata["Points"] to QList<QPointF>
        f.namedPoints(); // returns [QPointF(1, 1), QPointF(2, 2)]


## [QList][QList]&lt;[QPointF][QPointF]&gt; points() {: #points }

Get values stored in the [metadata](members.md#m_metadata) with key "Points". It is expected that this field holds a [QList][QList]&lt;[QPointf][QPointF]>&gt;.

* **function definition:**

        QList<QPointF> points() const

* **parameters:** NONE
* **output:** ([QList][QList]&lt;[QPointf][QPointF]>&gt;) Returns a list of points stored at [metadata](members.md#m_metadata)["Points"]
* **see:** [appendPoint](#appendpoint), [appendPoints](#appendpoints), [clearPoints](#clearpoints), [setPoints](#setpoints)
* **example:**

        File file;
        file.set("Points", QVariant::fromValue<QPointF>(QPointF(1, 1)));
        file.points(); // returns [] (point is not in a list)

        file.setPoints(QList<QPointF>() << QPointF(2, 2));
        file.points(); // returns [QPointF(2, 2)]


## void appendPoint(const [QPointF][QPointF] &point) {: #appendpoint }

Append a point to the [QList][QList]&lt;[QPointF][QPointF]&gt; stored at [metadata](members.md#m_metadata)["Points"].

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


## void appendPoints(const [QList][QList]&lt;[QPointF][QPointF]&gt; &points) {: #appendpoints }

Append a list of points to the [QList][QList]&lt;[QPointF][QPointF]&gt; stored at [metadata](members.md#m_metadata)["Points"].

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


## void clearPoints() {: #clearpoints }

Remove all points stored at [metadata](members.md#m_metadata)["Points"].

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


## void setPoints(const [QList][QList]&lt;[QPointF][QPointF]&gt; &points) {: #setpoints }

Replace the points stored at [metadata](members.md#m_metadata)["Points"]

* **function definition:**

        inline void setPoints(const QList<QPointF> &points)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    points | const [QList][QList]&lt;[QPointF][QPointF]&gt; & | Points to overwrite [metadata](members.md#m_metadata) with

* **output:** (void)
* **example:**

        File file;
        file.appendPoints(QList<QPointF>() << QPointF(1, 1) << QPointF(2, 2));
        file.points(); // returns [QPointF(1, 1), QPointF(2, 2)]

        file.setPoints(QList<QPointF>() << QPointF(3, 3) << QPointF(4, 4));
        file.points(); // returns [QPointF(3, 3), QPointF(4, 4)]


## [QList][QList]&lt;[QRectF][QRectF]&gt; namedRects() {: #namedrects }

Find values in the [metadata](members.md#m_metadata) that can be converted into [QRectF][QRectF]'s. Values stored as [QList][QList]&lt;[QRectF][QRectF]&gt; *will not** be returned.

* **function definition:**

        QList<QRectF> namedRects() const

* **parameters:** NONE
* **output:** ([QList][QList]&lt;[QRectF][QRectF]&gt;) Returns a list of rects that can be converted from [metadata](members.md#m_metadata) values.
* **example:**

        File file;
        file.set("Key1", QVariant::fromValue<QRectF>(QRectF(1, 1, 5, 5)));
        file.set("Key2", QVariant::fromValue<QRectF>(QRectF(2, 2, 5, 5)));
        file.set("Rects", QVariant::fromValue<QRectF>(QRectF(3, 3, 5, 5)));

        f.namedRects(); // returns [QRectF(1, 1, 5x5), QRectF(2, 2, 5x5), QRectF(3, 3, 5x5)]

        file.setRects(QList<QRectF>() << QRectF(3, 3, 5x5)); // changes metadata["Rects"] to QList<QRectF>
        f.namedRects(); // returns [QRectF(1, 1, 5x5), QRectF(2, 2, 5x5)]


## [QList][QList]&lt;[QRectF][QRectF]&gt; rects() {: #rects }

Get values stored at [metadata](members.md#m_metadata)["Rects"]. It is expected that this field holds a [QList][QList]&lt;[QRectf][QRectF]>&gt;.

* **function definition:**

        QList<QRectF> points() const

* **parameters:** NONE
* **output:** ([QList][QList]&lt;[QRectf][QRectF]>&gt;) Returns a list of rects stored at [metadata](members.md#m_metadata)["Rects"]
* **see:** [appendRect](#appendrect-1), [appendRects](#appendrects-1), [clearRects](#clearrects), [setRects](#setrects-1)
* **example:**

        File file;
        file.set("Rects", QVariant::fromValue<QRectF>(QRectF(1, 1, 5, 5)));
        file.rects(); // returns [] (rect is not in a list)

        file.setRects(QList<QRectF>() << QRectF(2, 2, 5, 5));
        file.rects(); // returns [QRectF(2, 2, 5x5)]


## void appendRect(const [QRectF][QRectF] &rect) {: #appendrect-1 }

Append a rect to the [QList][QList]&lt;[QRectF][QRectF]&gt; stored at [metadata](members.md#m_metadata)["Rects"].

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


## void appendRect(const [Rect][Rect] &rect) {: #appendrect-2 }

Append an OpenCV-style [Rect][Rect] to the [QList][QList]&lt;[QRectF][QRectF]&gt; stored at [metadata](members.md#m_metadata)["Rects"]. Supplied OpenCV-style rects are converted to [QRectF][QRectF] before being appended.

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


## void appendRects(const [QList][QList]&lt;[QRectF][QRectF]&gt; &rects) {: #appendrects-1 }

Append a list of rects to the [QList][QList]&lt;[QRectF][QRectF]&gt; stored at [metadata](members.md#m_metadata)["Rects"].

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


## void appendRects(const [QList][QList]&lt;[QRectF][QRectF]&gt; &rects) {: #appendrects-2 }

Append a list of OpenCV-style [Rects][Rect] to the [QList][QList]&lt;[QRectF][QRectF]&gt; stored at [metadata](members.md#m_metadata)["Rects"]. Supplied OpenCV-style rects are converted to [QRectF][QRectF] before being appended.

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

## void clearRects() {: #clearrects }

Remove all points stored at [metadata](members.md#m_metadata)["Rects"].

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


## void setRects(const [QList][QList]&lt;[QRectF][QRectF]&gt; &rects) {: #setrects-1 }

Replace the rects stored at [metadata](members.md#m_metadata) with a provided list of rects.

* **function definition:**

        inline void setRects(const QList<QRectF> &rects)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    rects | const [QList][QList]&lt;[QRectF][QRectF]&gt; & | Rects to overwrite [metadata](members.md#m_metadata)["Rects"] with

* **output:** (void)
* **example:**

        File file;
        file.appendRects(QList<QRectF>() << QRectF(1, 1, 5, 5) << QRectF(2, 2, 5, 5));
        file.rects(); // returns [QRectF(1, 1, 5x5), QRectF(2, 2, 5x5)]

        file.setRects(QList<QRectF>() << QRectF(3, 3, 5, 5) << QRectF(4, 4, 5, 5));
        file.rects(); // returns [QRectF(3, 3, 5x5), QRectF(4, 4, 5x5)]


## void setRects(const [QList][QList]&lt;[Rect][Rect]&gt; &rects) {: #setrects-2 }

Replace the rects stored at [metadata](members.md#m_metadata) with a provided list of OpenCV-style [Rects][Rect].

* **function definition:**

        inline void setRects(const QList<Rect> &rects)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    rects | const [QList][QList]&lt;[Rect][Rect]&gt; & | OpenCV-style rects to overwrite [metadata](members.md#m_metadata)["Rects"] with

* **output:** (void)
* **example:**

        File file;
        file.appendRects(QList<cv::Rect>() << cv::Rect(1, 1, 5, 5) << cv::Rect(2, 2, 5, 5));
        file.rects(); // returns [QRectF(1, 1, 5x5), QRectF(2, 2, 5x5)]

        file.setRects(QList<cv::Rect>() << cv::Rect(3, 3, 5, 5) << cv::Rect(4, 4, 5, 5));
        file.rects(); // returns [QRectF(3, 3, 5x5), QRectF(4, 4, 5x5)]

<!-- Links -->
[Qt]: http://qt-project.org/ "Qt"
[QApplication]: http://doc.qt.io/qt-5/qapplication.html "QApplication"
[QCoreApplication]: http://doc.qt.io/qt-5/qcoreapplication.html "QCoreApplication"
[QObject]: http://doc.qt.io/qt-5/QObject.html "QObject"
[Qt Property System]: http://doc.qt.io/qt-5/properties.html "Qt Property System"

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

[R]: http://www.r-project.org/ "R"
[Mat]: http://docs.opencv.org/modules/core/doc/basic_structures.html#mat "Mat"
[Rect]: http://docs.opencv.org/modules/core/doc/basic_structures.html#rect "Rect"
[InputArray]: http://docs.opencv.org/modules/core/doc/basic_structures.html#inputarray "InputArray"
[OutputArray]: http://docs.opencv.org/modules/core/doc/basic_structures.html#outputarray "OutputArray"
[OpenCV Image Formats]: http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=imread#imread "OpenCV Image Formats"
