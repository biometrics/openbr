<!-- FACTORY -->

For run time construction of objects from strings.

Uses the Industrial Strength Pluggable Factory model described [here](http://adtmag.com/articles/2000/09/25/industrial-strength-pluggable-factories.aspx).

OpenBR's plugin architecture is premised on the idea that algorithms can be described as strings and can be built at runtime. Constructing plugins from strings is the job of the [Factory](#factory). For a plugin to be built by the [Factory](#factory) it must inherit from [Object](#object). It also must be registered with the factory at compile time using [BR_REGISTER](#factory-macros-br_register). At runtime, the [Factory](#factory) will look up provided strings in its [registry](#factory-members-registry) and, if they exist, return the described plugins.


## Members {: #factory-members }

Member | Type | Description
--- | --- | ---
<a class="table-anchor" id=factory-members-registry></a>registry | static [QMap][QMap]&lt;[QString][QString],[Factory](#factory)&lt;<tt>T</tt>&gt;\*&gt; | List of all objects that have been registered with the factory. Registered objects are stored in this static registry by abstraction type.


## Constructors {: #factory-constructors }

Constructor \| Destructor | Description
--- | ---
Factory([QString][QString] name) | This is a special constructor in OpenBR. It is used to register new objects in the [registry](#factory-members-registry).
virtual ~Factory() | Default destructor

## Macros {: #factory-macros }

### BR_REGISTER {: #factory-macros-br_register }

A special macro to register plugins in the [Factory](#factory)::[registry](#factory-members-registry). When a plugin is registered the associated abstraction type will be removed from it's name, if it exists. For example, ```BR_REGISTER(Transform, ExampleTransform)``` will be registered as "Example". Plugins *do not* have to have the abstraction as part of their name.

* **macro definition:**

        #define BR_REGISTER(ABSTRACTION,IMPLEMENTATION)  

* **parameters:**

    Parameter | Description
    --- | ---
    ABSTRACTION | The Abstraction that the object inherits from. The object must inherit from [Object](#object) somewhere in its inheritance tree. Abstractions should also implement ```ABSTRACTION *make()```. See [Transform](#transform) as an example of an abstraction.
    IMPLEMENTATION | The Implementation of the object. This is the definition of the object you want returned when you call ```Factory<T>::make```.

* **example:**

        class Implementation : public Abstraction
        {
            Q_OBJECT

            ... // some functions etc.
        };

        BR_REGISTER(Abstraction, Implementation)


## Static Functions {: #factory-static-functions }

### <tt>T</tt> \*make(const [File](#file) &file) {: #factory-static-make }

This function constructs a plugin of type <tt>T</tt> from a provided [File](#file). The [File](#file) [name](#file-members-name) must already be in the [registry](#factory-members-registry) to be made.

* **function definition:**

        static T *make(const File &file)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    file | const [File](#file) & | File describing the object to be constructed

* **output:** (<tt>T</tt>) Returns an object of type <tt>T</tt>. <tt>T</tt> must inherit from [Object](#object).
* **example:**

        Transform *transform = Factory<Transform>::make("ExampleTransform(Property1=Value1,Property2=Value2)");
        // returns a pointer to an instance of ExampleTransform with property1 set to value1
        // and property2 set to value 2.

<!-- no italics* -->

### [QList][QList]&lt;[QSharedPointer][QSharedPointer]&lt;<tt>T</tt>&gt;&gt;makeAll {: factory-static-makeAll }

Make all of the registered plugins for a specific abstraction.

* **function definition:**

        static QList< QSharedPointer<T> > makeAll()

* **parameters:** NONE
* **output:** ([QList][QList]&lt;[QSharedPointer][QSharedPointer]&lt;<tt>T</tt>&gt;&gt;) Returns a list of all of the objects registered to a particular abstraction <tt>T</tt>
* **example:**

        BR_REGISTER(Transform, FirstTransform)
        BR_REGISTER(Transform, SecondTransform)

        QList<QSharedPointer<Transform> > = Factory<Transform>::makeAll(); // returns a list with pointers to FirstTransform and SecondTransform


### [QStringList][QStringList] names() {: #factory-static-names }

Get the names of all of the registered objects for a specific abstraction.

* **function definition:**

        static QStringList names()

* **parameters:** NONE
* **output:** ([QStringList][QStringList]) Returns a list of object names from the [registry](#factor-members-registry)
* **example:**

        BR_REGISTER(Transform, FirstTransform)
        BR_REGISTER(Transform, SecondTransform)

        QStringList names = Factory<Transform>::names(); // returns ["First", "Second"]


### [QString][QString] parameters(const [QString][QString] &name) {: #factory-static-parameters }

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
