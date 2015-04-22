<!-- UNTRAINABLE TRANSFORM -->

Inherits [Transform](../transform/transform.md)

A [Transform](../transform/transform.md) that does not require training.

See:

* [Members](members.md)
* [Constructors](constructors.md)
* [Static Functions](statics.md)
* [Functions](functions.md)

This is a base class for [Transforms](../transform/transform.md) that are not trainable. It overloads [train](../transform/functions.md#train-1), [load](../transform/functions.md#load), and [store](../transform/functions.md#store) so that they cannot be used in derived classes. Load and store are overloaded because a [Transform](../transform/transform.md) that doesn't train should also have nothing to save.