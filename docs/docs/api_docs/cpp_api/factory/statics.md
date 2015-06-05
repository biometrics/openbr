## <tt>T</tt> \*make(const [File](../file/file.md) &file) {: #make }

This function constructs a plugin of type <tt>T</tt> from a provided [File](../file/file.md). The [File](../file/file.md) [name](../file/members.md#name) must already be in the [registry](members.md#registry) to be made.

* **function definition:**

        static T *make(const File &file)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    file | const [File](../file/file.md) & | File describing the object to be constructed

* **output:** (<tt>T</tt>) Returns an object of type <tt>T</tt>. <tt>T</tt> must inherit from [Object](../object/object.md).
* **example:**

        Transform *transform = Factory<Transform>::make("ExampleTransform(Property1=Value1,Property2=Value2)");
        // returns a pointer to an instance of ExampleTransform with property1 set to value1
        // and property2 set to value 2.

## [QList][QList]&lt;[QSharedPointer][QSharedPointer]&lt;<tt>T</tt>&gt;&gt;makeAll {: #makeall }

Make all of the registered plugins for a specific abstraction.

* **function definition:**

        static QList< QSharedPointer<T> > makeAll()

* **parameters:** NONE
* **output:** ([QList][QList]&lt;[QSharedPointer][QSharedPointer]&lt;<tt>T</tt>&gt;&gt;) Returns a list of all of the objects registered to a particular abstraction <tt>T</tt>
* **example:**

        BR_REGISTER(Transform, FirstTransform)
        BR_REGISTER(Transform, SecondTransform)

        QList<QSharedPointer<Transform> > = Factory<Transform>::makeAll(); // returns a list with pointers to FirstTransform and SecondTransform


## [QStringList][QStringList] names() {: #names }

Get the names of all of the registered objects for a specific abstraction.

* **function definition:**

        static QStringList names()

* **parameters:** NONE
* **output:** ([QStringList][QStringList]) Returns a list of object names from the [registry](members.md#registry)
* **example:**

        BR_REGISTER(Transform, FirstTransform)
        BR_REGISTER(Transform, SecondTransform)

        QStringList names = Factory<Transform>::names(); // returns ["First", "Second"]


## [QString][QString] parameters(const [QString][QString] &name) {: #parameters }

Get the parameters for the plugin defined by the provided name.

* **function definition:**

        static QString parameters(const QString &name)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    name | const [QString][QString] & | Name of a plugin

* **output:** ([QString][QString]) Returns a string with each property and its value seperated by commas.
* **example:**

        class ExampleTransform : public Transform
        {
            Q_OBJECT

            Q_PROPERTY(int property1 READ get_property1 WRITE set_property1 RESET reset_property1 STORED false)
            Q_PROPERTY(float property2 READ get_property2 WRITE set_property2 RESET reset_property2 STORED false)
            Q_PROPERTY(QString property3 READ get_property3 WRITE set_property3 RESET reset_property3 STORED false)
            BR_PROPERTY(int, property1, 1)
            BR_PROPERTY(float, property2, 2.5)
            BR_PROPERTY(QString, property3, "Value")

            ...
        };

        Factory<Transform>::parameters("Example"); // returns "int property1 = 1, float property2 = 2.5, QString property3 = Value"
        Factory<Transform>::parameters("Example(property3=NewValue)"); // returns "int property1 = 1, float property2 = 2.5, QString property3 = NewValue"

<!-- Links -->
[QString]: http://doc.qt.io/qt-5/QString.html "QString"
[QStringList]: http://doc.qt.io/qt-5/qstringlist.html "QStringList"
[QList]: http://doc.qt.io/qt-5/QList.html "QList"
[QSharedPointer]: http://doc.qt.io/qt-5/qsharedpointer.html "QSharedPointer"
