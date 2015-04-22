<!-- UNTRAINABLE DISTANCE -->

Inherits [Distance](../distance/distance.md)

A [Distance](../distance/distance.md) that does not require training.

See:

* [Members](members.md)
* [Constructors](constructors.md)
* [Static Functions](statics.md)
* [Functions](functions.md)

This is a base class for [Distances](../distance/distance.md) that are not trainable. It overloads [trainable](../distance/functions.md#trainable) to return false and [train](../distance/functions.md#train) so that it cannot be used in derived classes.
