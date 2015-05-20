## BR_PROPERTY {: #br_property }

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

<!-- Links -->
[Qt Property System]: http://doc.qt.io/qt-5/properties.html "Qt Property System"
