Inherits [Distance](../distance/distance.md)

A [Distance](../distance/distance.md) with a list of child distances. The ListDistance is trainable if its children are trainable.

See:

* [Properties](properties.md)

This is a base class for [Distances](../distance/distance.md) that have a list of child distances. It overloads [trainable](../distance/functions.md#trainable) to return true if any of its children are trainable and false otherwise.
