## [Classifier](classifier.md) \*make([QString][QString] str, [QObject][QObject] \*parent) {: #make }

Make a [Classifier](classifier.md) from a string. The string is passed to [Factory](../factory/factory.md)::[make](../factory/statics.md#make) to be turned into a classifier.

* **function definition:**

        static Classifier *make(QString str, QObject *parent)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    str | [QString][QString] | String describing the classifier
    parent | [QObject][QObject] \* | Parent of the object to be created

* **output:** ([Classifier](classifier.md) \*) Returns a pointer to the [Classifier](classifier.md) described by the string
* **see:** [Factory::make](../factory/statics.md#make)
* **example:**

        Classifier *classifier = Classifier::make("Classifier(representation=Representation(property1=value1)");
        classifier->description(); // Returns "Classifier(representation=Representation(property1=value1))"

<!-- Links -->
[QString]: http://doc.qt.io/qt-5/QString.html "QString"
[QObject]: http://doc.qt.io/qt-5/QObject.html "QObject"
