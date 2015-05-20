## [Representation](representation.md) \*make([QString][QString] str, [QObject][QObject] \*parent) {: #make }

Make a [Representation](representation.md) from a string. The string is passed to [Factory](../factory/factory.md)::[make](../factory/statics.md#make) to be turned into a representation.

* **function definition:**

        static Representation *make(QString str, QObject *parent)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    str | [QString][QString] | String describing the representation
    parent | [QObject][QObject] \* | Parent of the object to be created

* **output:** ([Representation](representation.md) \*) Returns a pointer to the [Representation](representation.md) described by the string
* **see:** [Factory::make](../factory/statics.md#make)
* **example:**

        Representation *rep = Representation::make("Representation(property1=value1)");
        rep->description(); // Returns "Representation(property1=value1)"

<!-- Links -->
[QString]: http://doc.qt.io/qt-5/QString.html "QString"
[QObject]: http://doc.qt.io/qt-5/QObject.html "QObject"
