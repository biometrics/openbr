## [QVariant][QVariant] parse(const [QString][QString] &value) {: #parse }

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


## [QList][QList]&lt;[QVariant][QVariant]&gt; values(const [QList][QList]&lt;U&gt; &fileList, const [QString][QString] &key) {: #values }

Gather a list of [QVariant][QVariant] values associated with a metadata key from a provided list of files.

* **function definition:**

        template<class U> static [QList<QVariant> values(const QList<U> &fileList, const QString &key)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    fileList | const [QList][QList]&lt;U&gt; & | A list of files to parse for values. A type is required for <tt>U</tt>. Valid options are: <ul> <li>[File](file.md)</li> <li>[QString][QString]</li> </ul>
    key | const [QString][QString] & | A metadata key used to lookup the values.

* **output:** ([QList][QList]&lt;[QVariant][QVariant]&gt;) A list of [QVariant][QVariant] values associated with the given key in each of the provided files.
* **example:**

        File f1, f2;
        f1.set("Key1", QVariant::fromValue<float>(1));
        f1.set("Key2", QVariant::fromValue<float>(2));
        f2.set("Key1", QVariant::fromValue<float>(3));

        File::values<File>(QList<File>() << f1 << f2, "Key1"); // returns [QVariant(float, 1),
                                                               //          QVariant(float, 3)]


## [QList][QList]&lt;T&gt; get(const [QList][QList]&lt;U&gt; &fileList, const [QString][QString] &key) {: #get-1 }

Gather a list of <tt>T</tt> values associated with a metadata key from a provided list of files. <tt>T</tt> is a user provided type. If the key does not exist in the metadata of *any* file an error is thrown.

* **function definition:**

        template<class T, class U> static QList<T> get(const QList<U> &fileList, const QString &key)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    fileList | const [QList][QList]&lt;U&gt; & | A list of files to parse for values. A type is required for U. Valid options are: <ul> <li>[File](file.md)</li> <li>[QString][QString]</li> </ul>
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


## [QList][QList]&lt;T&gt; get(const [QList][QList]&lt;U&gt; &fileList, const [QString][QString] &key, const T &defaultValue) {: #get-2 }

Gather a list of <tt>T</tt> values associated with a metadata key from a provided list of files. <tt>T</tt> is a user provided type. If the key does not exist in the metadata of *any* file the provided **defaultValue** is used.

* **function definition:**

        template<class T, class U> static QList<T> get(const QList<U> &fileList, const QString &key, const T &defaultValue)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    fileList | const [QList][QList]&lt;U&gt; & | A list of files to parse for values. A type is required for U. Valid options are: <ul> <li>[File](file.md)</li> <li>[QString][QString]</li> </ul>
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


## [QDebug][QDebug] operator <<([QDebug][QDebug] dbg, const [File](file.md) &file) {: #dbg-operator-ltlt }

Calls [flat](functions.md#flat) on the given file and then streams that file to stderr.

* **function definition:**

        QDebug operator <<(QDebug dbg, const File &file)

* **parameter:**

    Parameter | Type | Description
    --- | --- | ---
    dbg | [QDebug][QDebug] | The debug stream
    file | const [File](file.md) & | File to stream

* **output:** ([QDebug][QDebug] &) returns a reference to the updated debug stream
* **example:**

        File file("../path/to/pictures/picture.jpg");
        file.set("Key", QString("Value"));

        qDebug() << file; // "../path/to/pictures/picture.jpg[Key=Value]" streams to stderr


## [QDataStream][QDataStream] &operator <<([QDataStream][QDatastream] &stream, const [File](file.md) &file) {: #stream-operator-ltlt }

Serialize a file to a data stream.

* **function definition:**

        QDataStream &operator <<(QDataStream &stream, const File &file)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    stream | [QDataStream][QDataStream] | The data stream
    file | const [File](file.md) & | File to stream

* **output:** ([QDataStream][QDataStream] &) returns a reference to the updated data stream
* **example:**

        void store(QDataStream &stream)
        {
            File file("../path/to/pictures/picture.jpg");
            file.set("Key", QString("Value"));

            stream << file; // "../path/to/pictures/picture.jpg[Key=Value]" serialized to the stream
        }


## [QDataStream][QDataStream] &operator >>([QDataStream][QDataStream] &stream, const [File](file.md) &file) {: #stream-operator-gtgt }

Deserialize a file from a data stream.

* **function definition:**

        QDataStream &operator >>(QDataStream &stream, const File &file)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    stream | [QDataStream][QDataStream] | The data stream
    file | const [File](file.md) & | File to stream

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

<!-- Links -->
[QList]: http://doc.qt.io/qt-5/QList.html "QList"
[QVariant]: http://doc.qt.io/qt-5/qvariant.html "QVariant"
[QString]: http://doc.qt.io/qt-5/QString.html "QString"
[QDebug]: http://doc.qt.io/qt-5/qdebug.html "QDebug"
[QDataStream]: http://doc.qt.io/qt-5/qdatastream.html "QDataStream"
[QRectF]: http://doc.qt.io/qt-5/qrectf.html "QRectF"
[QPointF]: http://doc.qt.io/qt-5/qpointf.html "QPointF"
