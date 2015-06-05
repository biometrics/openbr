<!-- UNTRAINABLE TRANSFORM -->

Inherits [Transform](../transform/transform.md)

A [Transform](../transform/transform.md) that does not require training.

See:

* [Constructors](constructors.md)

This is a base class for [Transforms](../transform/transform.md) that are not trainable. It overloads [train](../transform/functions.md#train-1), [load](../object/functions.md#load), and [store](../object/functions.md#store) so that they cannot be used in derived classes. Load and store are overloaded because a [Transform](../transform/transform.md) that doesn't train should also have nothing to save.
