<!-- OBJECT -->

Inherits from [QObject][QObject]

This is the base class of all OpenBR plugins. [Objects](#object) are constructed from [Files](#files). The [File's](#file) [name](#file-members-name) specifies which plugin to construct and the [File's](#file) [metadata](#file-members-m_metadata) provides initialization values for the plugin's properties.

## Members {: #object-members }

Member | Type | Description
--- | --- | ---
file | [File](#file) | The [File](#file) used to construct the plugin.
firstAvailablePropertyIdx | int | Index of the first property of the object that can be set via command line arguments

## Macros {: #object-macros }

### BR_PROPERTY {: #object-macros-br_property }

This macro provides an extension to the [Qt Property System][Qt Property System]. It's purpose is to set default values for each property in an object. Every call to <tt>BR_PROPERTY</tt> should have a corresponding call to <tt>Q_PROPERTY</tt>.

* **macro definition:**

        #define BR_PROPERTY(TYPE,NAME,DEFAULT)

* **parameters:**

    Parameter | Description
    --- | ---
    TYPE | The type of the property (int, float etc.)
    NAME | The name of the property
    DEFAULT | The default value of the property

* **example:**

        class ExampleTransform : public Transform
        {
            Q_OBJECT

            Q_PROPERTY(int property1 READ get_property1 WRITE set_property1 RESET reset_property1 STORED false)
            Q_PROPERTY(float property2 READ get_property2 WRITE set_property2 RESET reset_property2 STORED false)
            Q_PROPERTY(QString property3 READ get_property3 WRITE set_property3 RESET reset_property3 STORED false)
            BR_PROPERTY(int, property1, 1) // sets default value of "property1" to 1
            BR_PROPERTY(float, property2, 2.5) // sets default value of "property2" to 2.5
            BR_PROPERTY(QString, property3, "Value") // sets default value of "property3" to "Value"

            ...
        };


## Static Functions {: #object-static-functions }

### [QStringList][QStringList] parse(const [QString][QString] &string, char split = ',')

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


## Functions {: #object-functions }

### void init() {: #object-function-init }

This is a virtual function. It is meant to be overloaded by derived classes if they need to initialize internal variables. The default constructor of derived objects should *never* be overloaded.

* **function definition:**

        virtual void init()

* **parameters:** NONE
* **output:** (void)


### void store([QDataStream][QDataStream] &stream)

This is a virtual function. Serialize an item to a [QDataStream][QDataStream]. The default implementation serializes each property and its value to disk.

* **function definition:**

        virtual void store(QDataStream &stream) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    stream | [QDataStream][QDataStream] & | Stream to store serialized data
