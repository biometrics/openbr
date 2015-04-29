<!-- FACTORY -->

For the run time construction of objects from strings.

See:

* [Members](members.md)
* [Constructors](constructors.md)
* [Macros](macros.md)
* [Static Functions](statics.md)

Uses the Industrial Strength Pluggable Factory model described [here](http://adtmag.com/articles/2000/09/25/industrial-strength-pluggable-factories.aspx).

OpenBR's plugin architecture is premised on the idea that algorithms can be described as strings and can be built at runtime. Constructing plugins from strings is the job of the [Factory](factory.md). For a plugin to be built by the [Factory](factory.md) it must inherit from [Object](../object/object.md). It also must be registered with the factory at compile time using [BR_REGISTER](macros.md#br_register). At runtime, the [Factory](factory.md) will look up provided strings in its [registry](members.md#registry) and, if they exist, return the described plugins.
