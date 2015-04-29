<!-- INITIALIZER -->

Inherits [Object](../object/object.md)

See:

* [Constructors](constructors.md)
* [Functions](functions.md)

Plugin base class for initializing resources. On startup (the call to [Context](../context/context.md)::[initialize](../context/statics.md#initialize)), OpenBR will call [initialize](functions.md#initialize) on every Initializer that has been registered with the [Factory](../factory/factory.md). On shutdown (the call to [Context](../context/context.md)::[finalize](../context/statics.md#finalize), OpenBR will call [finalize](functions.md#finalize) on every registered initializer.

The general use case for initializers is to launch shared contexts for third party integrations into OpenBR. These cannot be launched during [Transform](../transform/transform.md)::[init](../object/functions.md#init) for example, because multiple instances of the [Transform](../transform/transform.md) object could exist across multiple threads.
