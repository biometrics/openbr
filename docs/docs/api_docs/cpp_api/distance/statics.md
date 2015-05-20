## [Distance](distance.md) \*make([QString][QString] str, [QObject][QObject] \*parent) {: #make }

Make a [Distance](distance.md) from a string. This function converts the abbreviation character **+** into it's full-length alternative.

Abbreviation | Translation
--- | ---
\+ | [PipeDistance](../../../plugin_docs/distance.md#pipedistance). Each [Distance](distance.md) linked by a **+** is turned into a child of a single [PipeDistance](../../../plugin_docs/distance.md#pipedistance). "Distance1+Distance2" becomes "Pipe([Distance1,Distance2])". [Templates](../template/template.md) are projected through the children of a pipe in series, the output of one become the input of the next.

The expanded string is then passed to [Factory](../factory/factory.md)::[make](../factory/statics.md#make) to be turned into a distance.

* **function definition:**

        static Distance *make(QString str, QObject *parent)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    str | [QString][QString] | String describing the distance
    parent | [QObject][QObject] \* | Parent of the object to be created

* **output:** ([Distance](distance.md) \*) Returns a pointer to the [Distance](distance.md) described by the string
* **see:** [Factory::make](../factory/statics.md#make)
* **example:**

        Distance::make("Distance1+Distance2+Distance3")->description(); // returns "Pipe(distances=[Distance1,Distance2,Distance3])".

## [QSharedPointer][QSharedPointer]&lt;[Distance](distance.md)&gt; fromAlgorithm(const [QString][QString] &algorithm) {: #fromalgorithm }

Create a [Distance](distance.md) from an OpenBR algorithm string. The [Distance](distance.md) is created using everything to the right of a **:** or a **!** in the string.

* **function definition:**

        static QSharedPointer<Distance> fromAlgorithm(const QString &algorithm)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    algorithm | const [QString][QString] & | Algorithm string to construct the [Distance](distance.md) from

* **output:** ([QSharedPointer][QSharedPointer]&lt;[Distance](distance.md)&gt;) Returns a pointer to the [Distance](distance.md) described by the algorithm.
* **example:**

    Distance::fromAlgorithm("EnrollmentTransform:Distance")->decription(); // returns "Distance"
    Distance::fromAlgorithm("EnrollmentTransform!Distance1+Distance2")->decription(); // returns "Pipe(distances=[Distance1,Distance2])

<!-- Links -->
[QString]: http://doc.qt.io/qt-5/QString.html "QString"
[QObject]: http://doc.qt.io/qt-5/QObject.html "QObject"
[QSharedPointer]: http://doc.qt.io/qt-5/qsharedpointer.html "QSharedPointer"
