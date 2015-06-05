## BR_REGISTER {: #br_register }

A special macro to register plugins in the [Factory](factory.md)::[registry](members.md#registry). When a plugin is registered the associated abstraction type will be removed from it's name, if it exists. For example, ```BR_REGISTER(Transform, ExampleTransform)``` will be registered as "Example". Plugins *do not* have to have the abstraction as part of their name.

* **macro definition:**

        #define BR_REGISTER(ABSTRACTION,IMPLEMENTATION)  

* **parameters:**

    Parameter | Description
    --- | ---
    ABSTRACTION | The Abstraction that the object inherits from. The object must inherit from [Object](../object/object.md) somewhere in its inheritance tree. Abstractions should also implement ```ABSTRACTION *make()```. See [Transform](../transform/transform.md) as an example of an abstraction.
    IMPLEMENTATION | The Implementation of the object. This is the definition of the object you want returned when you call ```Factory<T>::make```.

* **example:**

        class Implementation : public Abstraction
        {
            Q_OBJECT

            ... // some functions etc.
        };

        BR_REGISTER(Abstraction, Implementation)
