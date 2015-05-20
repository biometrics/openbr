## [QStringList][QStringList] parse(const [QString][QString] &string, char split = ',') {: #parse }

Split the provided string using the provided split character. Lexical scoping of <tt>()</tt>, <tt>[]</tt>, <tt>\<\></tt>, and <tt>{}</tt> is respected.

* **function definition:**

        static QStringList parse(const QString &string, char split = ',');

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    string | const [QString][QString] & | String to be split
    split | char | (Optional) The character to split the string on. Default is ','

* **output:** ([QStringList][QStringList]) Returns a list of the split strings
* **example:**

        Object::parse("Transform1(p1=v1,p2=v2),Transform2(p1=v3,p2=v4)"); // returns ["Transform1(p1=v1,p2=v2)", "Transform2(p1=v3,p2=v4)"]

<!-- Links -->
[QString]: http://doc.qt.io/qt-5/QString.html "QString"
[QStringList]: http://doc.qt.io/qt-5/qstringlist.html "QStringList"
