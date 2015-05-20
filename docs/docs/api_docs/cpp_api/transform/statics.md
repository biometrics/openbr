## [Transform](transform.md) \*make([QString][QString] str, [QObject][QObject] \*parent) {: #make }

Make a transform from a string. This function converts the abbreviation characters **+**, **/**, **{}**, **<\>**, and **()** into their full-length alternatives.

Abbreviation | Translation
--- | ---
\+ | [PipeTransform](../../../plugin_docs/core.md#pipetransform). Each [Transform](transform.md) linked by a **+** is turned into a child of a single [PipeTransform](../../../plugin_docs/core.md#pipetransform). "Example1+Example2" becomes "Pipe([Example1,Example2])". [Templates](../template/template.md) are projected through the children of a pipe in series, the output of one become the input of the next.
/ | [ForkTransform](../../../plugin_docs/core.md#forktransform). Each [Transform](transform.md) linked by a **/** is turned into a child of a single [ForkTransform](../../../plugin_docs/core.md#forktransform). "Example1/Example2" becomes "Fork([Example1,Example2])". [Templates](../template/template.md) are projected the children of a fork in parallel, each receives the same input and the outputs are merged together.
\{\} | [CacheTransform](../../../plugin_docs/core.md#cachetransform). Can only surround a single [Transform](transform.md). "{Example}" becomes "Cache(Example)". The results of a cached [Transform](transform.md) are stored in a global cache using the [file](../object/members.md#file) name as a key.
<> | [LoadStoreTransform](../../../plugin_docs/core.md#loadstoretransform). Can only surround a single [Transform](transform.md). "<Example>" becomes "LoadStore(Example)". Serialize and store a [Transform](transform.md) after training or deserialize and load a [Transform](transform.md) before projecting.
() | Order of operations. Change the order of operations using parantheses.

The parsed string is then passed to [Factory](../factory/factory.md)::[make](../factory/statics.md#make) to be turned into a transform.

* **function definition:**

        static Transform *make(QString str, QObject *parent)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    str | [QString][QString] | String describing the transform
    parent | [QObject][QObject] \* | Parent of the object to be created

* **output:** ([Transform](transform.md) \*) Returns a pointer to the [Transform](transform.md) described by the string
* **see:** [Factory::make](../factory/statics.md#make)
* **example:**

        Transform::make("Example1+Example2+<ModelFile>")->description(); // returns "Pipe(transforms=[Example1,Example2,LoadStore(ModelFile)])".


## [QSharedPointer][QSharedPointer]&lt;[Transform](transform.md)&gt; fromAlgorithm(const [QString][QString] &algorithm, bool preprocess=false) {: #fromalgorithm }

Create a [Transform](transform.md) from an OpenBR algorithm string. The [Transform](transform.md) is created using everything to the left of a **:** or a **!** in the string.

* **function definition:**

        static QSharedPointer<Transform> fromAlgorithm(const QString &algorithm, bool preprocess=false)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    algorithm | const [QString][QString] & | Algorithm string to construct the [Transform](transform.md) from
    preprocess | bool | (Optional) If true add a [StreamTransform](../../../plugin_docs/core.md#streamtransform) as the parent of the constructed [Transform](transform.md). Default is false.

* **output:** ([QSharedPointer][QSharedPointer]&lt;[Transform](transform.md)&gt;) Returns a pointer to the [Transform](transform.md) described by the algorithm.
* **example:**

    Transform::fromAlgorithm("EnrollmentTransform:Distance")->decription(); // returns "EnrollmentTransform"
    Transform::fromAlgorithm("EnrollmentTransform!DistanceTransform")->decription(); // returns "EnrollmentTransform"
    Transform::fromAlgorithm("EnrollmentTransform")->decription(); // returns "EnrollmentTransform"

## [QSharedPointer][QSharedPointer]<[Transform](transform.md)> fromComparison(const [QString][QString] &algorithm) {: #fromcomparison }

Create a[Transform](transform.md) from an OpenBR algorithm string. The [Transform](transform.md) is created using everything to the right of a **:** or a **!** in the string. If the separating symbol is a **:** the string to the right describes a distance. It is converted to a [GalleryCompareTransform](../../../plugin_docs/core.md#gallerycomparetransform) with the distance stored as a property. If the separating symbol is a **!** the string already describes a transform and is unchanged.

* **function definition:**

        static QSharedPointer<Transform> fromComparison(const QString &algorithm)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    algorithm | const [QString][QString] & | Algorithm string to construct the [Transform](transform.md) from

* **output:** ([QSharedPointer][QSharedPointer]&lt;[Transform](transform.md)&gt;) Returns a pointer to the [Transform](transform.md) described by the algorithm.
* **example:**

        Transform::fromAlgorithm("EnrollmentTransform:Distance")->description(); // returns "GalleryCompare(distance=Distance)""
        Transform::fromAlgorithm("EnrollmentTransform!DistanceTransform"); // returns "DistanceTransform"


## [Transform](transform.md) \*deserialize([QDataStream][QDataStream] &stream) {: #deserialize }

Deserialize a [Transform](transform.md) from a stream.

* **function definition:**

        static Transform *deserialize(QDataStream &stream)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    stream | [QDataStream][QDataStream] & | Stream containing the serialized transform

* **output:** ([Transform](transform.md) \*) Returns the deserialized transform


## [Template](../template/template.md) &operator&gt;&gt;([Template](../template/template.md) &srcdst, const [Transform](transform.md) &f) {: #template-operater-gtgt-1 }

Convenience function for [project](functions.md#project-1)

* **function definition:**

        inline Template &operator>>(Template &srcdst, const Transform &f)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    srcdst | [Template](../template/template.md) & | Input template. Will be overwritten with the output following call to [project](functions.md#project-1)
    f | const [Transform](transform.md) & | [Transform](transform.md) to project through.

* **output:** ([Template](../template/template.md) &) Returns the output of f::[project](functions.md#project-1)
* **example:**

        Template t("picture1.jpg");
        Transform *transform = Transform::make("Example", NULL);

        t >> *transform; // projects t through Example. t is overwritten with the output of the project call

<!--no italics* -->

## [TemplateList](../templatelist/templatelist.md) &operator&gt;&gt;([TemplateList](../templatelist/templatelist.md) &srcdst, const [Transform](transform.md) &f) {: #template-operater-gtgt-2 }

Convenience function for [project](functions.md#project-2)

* **function definition:**

        inline TemplateList &operator>>(TemplateList &srcdst, const Transform &f)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    srcdst | [TemplateList](../templatelist/templatelist.md) & | Input templates. Will be overwritten with the output following call to [project](functions.md#project-2)
    f | const [Transform](transform.md) & | [Transform](transform.md) to project through.

* **output:** ([TemplateList](../templatelist/templatelist.md) &) Returns the output of f::[project](functions.md#project-2)
* **example:**

        TemplateList tList(QList<Template>() << Template("picture1.jpg"));
        Transform *transform = Transform::make("Example", NULL);

        tList >> *transform; // projects tList through Example. tList is overwritten with the output of the project call

<!--no italics* -->

## [QDataStream][QDataStream] &operator&lt;&lt;([QDataStream][QDataStream] &stream, const [Transform](transform.md) &f) {: #stream-operator-ltlt}

Convenience function for [store](../object/functions.md#store)

* **function definition:**

        inline QDataStream &operator<<(QDataStream &stream, const Transform &f)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    stream | [QDataStream][QDataStream] & | Stream to store the transform in
    f | const [Transform](transform.md) & | Transform to be stored

* **output:** ([QDataStream][QDataStream] &) Returns the stream with the transform stored in it
* **example:**

        Transform *transform = Transform::make("Example(property1=value1,property2=value2)");

        QDataStream stream;
        stream << *transform; // stores "Example(property1=value1,property2=value2)" in the stream

<!--no italics* -->

## [QDataStream][QDataStream] &operator&gt;&gt;([QDataStream][QDataStream] &stream, const [Transform](transform.md) &f)  {: #stream-operator-gtgt}

Convenience function for [load](../object/functions.md#load)

* **function definition:**

        inline QDataStream &operator>>(QDataStream &stream, const Transform &f)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    stream | [QDataStream][QDataStream] & | Stream to load the transform from
    f | const [Transform](transform.md) & | Transform to store loaded information

* **output:** ([QDataStream][QDataStream] &) Returns the stream without the transform data
* **example:**

        Transform *in = Transform::make("Example(property1=value1,property2=value2)");

        QDataStream stream;
        stream << *in; // stores "Example(property1=value1,property2=value2)" in the stream

        Transform out;
        stream >> out;
        out->description(); // returns "Example(property1=value1,property2=value2)"

<!-- Links -->
[QString]: http://doc.qt.io/qt-5/QString.html "QString"
[QObject]: http://doc.qt.io/qt-5/QObject.html "QObject"
[QSharedPointer]: http://doc.qt.io/qt-5/qsharedpointer.html "QSharedPointer"
[QDataStream]: http://doc.qt.io/qt-5/qdatastream.html "QDataStream"
