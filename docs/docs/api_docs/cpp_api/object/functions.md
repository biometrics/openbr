
## void init() {: #init }

This is a virtual function. It is meant to be overloaded by derived classes if they need to initialize internal variables. The default constructor of derived objects should *never* be overloaded.

* **function definition:**

        virtual void init()

* **parameters:** NONE
* **output:** (void)

---

## void store([QDataStream][QDataStream] &stream) {: #store }

This is a virtual function. Serialize an object to a [QDataStream][QDataStream]. The default implementation serializes each property and its value to disk.

* **function definition:**

        virtual void store(QDataStream &stream) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    stream | [QDataStream][QDataStream] & | Stream to store serialized data

* **output:** (void)

---

## void load([QDataStream][QDataStream] &stream) {: #load }

This is a virtual function. Deserialize an item from a [QDataStream][QDataStream]. Elements can be deserialized in the same order in which they were serialized. The default implementation deserializes a value for each property and then calls [init](#init).

* **function definition:**

        virtual void load(QDataStream &stream);

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    stream | [QDataStream][QDataStream] & | Stream to deserialize data from

* **output:** (void)

---

## void serialize([QDataStream][QDataStream] &stream) {: #serialize }

This is a virtual function. Serialize an entire plugin to a [QDataStream][QDataStream]. This function is larger in scope then [store](#store). It stores the string describing the plugin and then calls [store](#store) to serialize its parameters. This has the benefit of being able to deserialize an entire plugin (or series of plugins) from a stored model file.

* **function definition:**

        virtual void serialize(QDataStream &stream) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    stream | [QDataStream][QDataStream] & | Stream to store serialized data

* **output:** (void)

---

## [QStringList][QStringList] parameters() {: #parameters }

Get a string describing the parameters of the object

* **function definition:**

        QStringList parameters() const

* **parameters:** NONE
* **output:** ([QStringList][QStringList]) Returns a list of the parameters to a function
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

        BR_REGISTER(Transform, ExampleTransform)

        Factory<Transform>::make(".Example")->parameters(); // returns ["int property1 = 1", "float property2 = 2.5", "QString property3 = Value"]

---

## [QStringList][QStringList] prunedArguments(bool expanded = false) {: #prunedarguments }

Get a string describing the user-specified parameters of the object. This means that parameters using their default value are not returned.

* **function definition:**

        QStringList prunedArguments(bool expanded = false) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    expanded | bool | (Optional) If true, expand all abbreviations or model files into their full description strings. Default is false.

* **output:** ([QStringList][QStringList]) Returns a list of all of the user specified parameters of the object
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

        BR_REGISTER(Transform, ExampleTransform)


        Factory<Transform>::make(".Example")->prunedArguments(); // returns []
        Factory<Transform>::make(".Example(property1=10)")->prunedArguments(); // returns ["property1=10"]
        Factory<Transform>::make(".Example(property1=10,property3=NewValue)")->prunedArguments(); // returns ["property1=10", "property3=NewValue"]

---

## [QString][QString] argument(int index, bool expanded) {: #argument }

Get a string value of the argument at a provided index. An index of 0 returns the name of the object.

* **function definition:**

        QString argument(int index, bool expanded) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    index | int | Index of the parameter to look up
    expanded | bool | If true, expand all abbreviations or model files into their full description strings.

* **output:** ([QString][QString]) Returns a string value for the lookup argument
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

        BR_REGISTER(Transform, ExampleTransform)

        Factory<Transform>::make(".Example")->argument(0, false); // returns "Example"
        Factory<Transform>::make(".Example")->argument(1, false); // returns "1"
        Factory<Transform>::make(".Example")->argument(2, false); // returns "2.5"

---

## [QString][QString] description(bool expanded = false) {: #description }

This is a virtual function. Get a description of the object. The description includes the name and any user-defined parameters of the object. Parameters that retain their default value are not included.

* **function definition:**

        virtual QString description(bool expanded = false) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    expanded | bool | (Optional) If true, expand all abbreviations or model files into their full description strings. Default is false.

* **output:** ([QString][QString]) Returns a string describing the object
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

        BR_REGISTER(Transform, ExampleTransform)

        Factory<Transform>::make(".Example")->description(); // returns "Example"
        qDebug() << Factory<Transform>::make(".Example(property3=NewValue)")->description(); // returns "Example(property3=NewValue)"

---

## void setProperty(const [QString][QString] &name, [QVariant][QVariant] value) {: #setproperty }

Set a property with a provided name to a provided value. This function overloads [QObject][QObject]::setProperty so that it can handle OpenBR data types. If the provided name is not a property of the object nothing happens. If the provided name is a property but the provided value is not a valid type an error is thrown.

* **function definition:**

        void setProperty(const QString &name, QVariant value)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    name | const [QString][QString] & | Name of the property to set
    value | [QVariant][QVariant] | Value to set the property to

* **output:** (void)
* **see:** [setPropertyRecursive](#setpropertyrecursive), [setExistingProperty](#setexistingproperty)
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

        BR_REGISTER(Transform, ExampleTransform)

        QScopedPointer<Transform> transform(Factory<Transform>::make(".Example"));
        transform->parameters(); // returns ["int property1 = 1", "float property2 = 2.5", "QString property3 = Value"]

        transform->setProperty("property1", QVariant::fromValue<int>(10));
        transform->parameters(); // returns ["int property1 = 10", "float property2 = 2.5", "QString property3 = Value"]

        transform->setProperty("property1", QVariant::fromValue<QString>("Value")); // ERROR: incorrect type

---

## bool setPropertyRecursive(const [QString][QString] &name, [QVariant][QVariant] value) {: #setpropertyrecursive }

Set a property of the object or the object's children to a provided value. The recursion is only single level; the children of the the objects children will not be affected. Only the first property found is set. This means that if a parent and a child have the same property only the parent's property is set.

* **function definition:**

        virtual bool setPropertyRecursive(const QString &name, QVariant value)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    name | const [QString][QString] & | Name of the property to set
    value | [QVariant][QVariant] | Value to set the property to

* **output:** (bool) Returns true if the property is set in either the object or its children
* **see:** [setProperty](#setproperty), [setExistingProperty](#setexistingproperty), [getChildren](#getchildren-2)
* **example:**

        class ChildTransform : public Transform
        {
            Q_OBJECT

            Q_PROPERTY(int property1 READ get_property1 WRITE set_property1 RESET reset_property1 STORED false)
            Q_PROPERTY(float property2 READ get_property2 WRITE set_property2 RESET reset_property2 STORED false)
            BR_PROPERTY(int, property1, 2)
            BR_PROPERTY(int, property2, 2.5)

            ...
        };

        BR_REGISTER(Transform, ChildTransform)

        class ParentTransform : public Transform
        {
            Q_OBJECT

            Q_PROPERTY(br::Transform *child READ get_child WRITE set_child RESET reset_child STORED false)
            Q_PROPERTY(int property1 READ get_property1 WRITE set_property1 RESET reset_property1 STORED false)
            BR_PROPERTY(br::Transform*, child, Factory<Transform>::make(".Child"))
            BR_PROPERTY(int, property1, 1)

            ...
        };

        QScopedPointer<Transform> parent(Factory<Transform>::make(".Parent"));
        parent->parameters(); // returns ["br::Transform* child = ", "int property1 = 1"]
        parent->getChildren<Transform>().first()->parameters(); // returns ["int property1 = 2", "float property2 = 2"]

        parent->setPropertyRecursive("property1", QVariant::fromValue<int>(10));
        parent->parameters(); // returns ["br::Transform* child = ", "int property1 = 10"]
        parent->getChildren<Transform>().first()->parameters(); // returns ["int property1 = 2", "float property2 = 2"]

        parent->setPropertyRecursive("property2", QVariant::fromValue<float>(10.5));
        parent->parameters(); // returns ["br::Transform* child = ", "int property1 = 10"]
        parent->getChildren<Transform>().first()->parameters(); // returns ["int property1 = 2", "float property2 = 10"]

---

## bool setExistingProperty(const [QString][QString] &name, [QVariant][QVariant] value) {: #setexistingproperty }

Attempt to set a property to a provided value. If the provided value is not a valid type for the given property an error is thrown.

* **function definition:**

        bool setExistingProperty(const QString &name, QVariant value)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    name | const [QString][QString] & | Name of the property to set
    value | [QVariant][QVariant] | Value to set the property to

* **output:** (bool) Returns true if the provided property exists and can be set to the provided value, otherwise returns false
* **example:**

        class ExampleTransform : public Transform
        {
            Q_OBJECT

            Q_PROPERTY(int property1 READ get_property1 WRITE set_property1 RESET reset_property1 STORED false)
            Q_PROPERTY(float property2 READ get_property2 WRITE set_property2 RESET reset_property2 STORED false)
            BR_PROPERTY(int, property1, 2)
            BR_PROPERTY(int, property2, 2.5)

            ...
        };

        BR_REGISTER(Transform, ExampleTransform)

        QScopedPointer<Transform> transform(Factory<Transform>::make(".Child"));
        transform->setExistingProperty("property1", QVariant::fromValue<int>(10)); // returns true
        transform->parameters(); // returns ["int property1 = 10", "float property2 = 2"]

        transform->setExistingProperty("property3", QVariant::fromValue<int>(10)); // returns false
        transform->setExistingProperty("property1", QVariant::fromValue<QString>("Hello")); // ERROR: incorrect type

---

## [QList][QList]&lt;[Object](object.md) \*&gt; getChildren() {: #getchildren-1 }

This is a virtual function. Get all of the children of the object. The default implementation looks for children in the properties of the object. A derived object should overload this function if it needs to provide children from a different source.

* **function definition:**

        virtual QList<Object *> getChildren() const

<!-- no italics* -->

* **parameters:** NONE
* **output:** ([QList][QList]&lt;[Object](object.md) \*&gt;) Returns a list of all of the children of the object
* **example:**

        class ChildTransform : public Transform
        {
            Q_OBJECT

            Q_PROPERTY(int property1 READ get_property1 WRITE set_property1 RESET reset_property1 STORED false)
            Q_PROPERTY(float property2 READ get_property2 WRITE set_property2 RESET reset_property2 STORED false)
            BR_PROPERTY(int, property1, 2)
            BR_PROPERTY(int, property2, 2.5)

            ...
        };

        BR_REGISTER(Transform, ChildTransform)

        class ParentTransform : public Transform
        {
            Q_OBJECT

            Q_PROPERTY(br::Transform *child READ get_child WRITE set_child RESET reset_child STORED false)
            Q_PROPERTY(int property1 READ get_property1 WRITE set_property1 RESET reset_property1 STORED false)
            BR_PROPERTY(br::Transform *, child, Factory<Transform>::make(".Child"))
            BR_PROPERTY(int, property1, 1)

            ...
        };

        BR_REGISTER(Transform, ParentTransform)

        QScopedPointer<Transform> transform(Factory<Transform>::make(".Parent"));
        transform->getChildren(); // returns [br::ChildTransform(0x7fc10bf01050, name = "Child")]
        transform->getChildren().first()->parameters(); // returns ["int property1 = 2", "float property2 = 2"]

---

## [QList][QList]&lt;T \*&gt; getChildren() {: #getchildren-2 }

Provides a wrapper on [getChildren](#getchildren-1) as a convenience to allow the return type (<tt>T</tt>) to be specified. <tt>T</tt> must be a derived class of [Object](object.md).

* **function definition:**

        template<typename T>
        QList<T *> getChildren() const

* **parameters:** NONE
* **output:** ([QList][QList]&lt;<tt>T</tt> \*&gt;) Returns a list of all of the children of the object, casted to type <tt>T</tt>. <tt>T</tt> must be a derived class of [Object](object.md)
* **example:**

        class ChildTransform : public Transform
        {
            Q_OBJECT

            Q_PROPERTY(int property1 READ get_property1 WRITE set_property1 RESET reset_property1 STORED false)
            Q_PROPERTY(float property2 READ get_property2 WRITE set_property2 RESET reset_property2 STORED false)
            BR_PROPERTY(int, property1, 2)
            BR_PROPERTY(int, property2, 2.5)

            ...
        };

        BR_REGISTER(Transform, ChildTransform)

        class ParentTransform : public Transform
        {
            Q_OBJECT

            Q_PROPERTY(br::Transform *child READ get_child WRITE set_child RESET reset_child STORED false)
            Q_PROPERTY(int property1 READ get_property1 WRITE set_property1 RESET reset_property1 STORED false)
            BR_PROPERTY(br::Transform *, child, Factory<Transform>::make(".Child"))
            BR_PROPERTY(int, property1, 1)

            ...
        };

        BR_REGISTER(Transform, ParentTransform)

        QScopedPointer<Transform> transform(Factory<Transform>::make(".Parent"));
        transform->getChildren<Transform>(); // returns [br::ChildTransform(0x7fc10bf01050, name = "Child")]
        transform->getChildren<Transform>().first()->parameters(); // returns ["int property1 = 2", "float property2 = 2"]

<!-- Links -->
[QDataStream]: http://doc.qt.io/qt-5/qdatastream.html "QDataStream"
[QString]: http://doc.qt.io/qt-5/QString.html "QString"
[QStringList]: http://doc.qt.io/qt-5/qstringlist.html "QStringList"
[QVariant]: http://doc.qt.io/qt-5/qvariant.html "QVariant"
[QObject]: http://doc.qt.io/qt-5/QObject.html "QObject"
[QList]: http://doc.qt.io/qt-5/QList.html "QList"
