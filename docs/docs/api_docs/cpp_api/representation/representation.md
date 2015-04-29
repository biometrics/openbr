<!-- REPRESENTATION -->

Inherits [Object](../object/object.md).

Plugin base class for converting images into feature vectors

See:

* [Constructors](constructors.md)
* [Static Functions](statics.md)
* [Functions](functions.md)

[Representations](representation.md) are used to convert images to feature vectors lazily (only when necessary). They are similar to [Transforms](../transform/transform.md) in many respects but differ in a few key areas. [Transforms](../transform/transform.md) should be used to construct feature vectors if it is desirable to construct a vector before evaluation that encompasses the entire feature space (or a smaller subset learned during training). [Representations](representation.md) should be used if their is a large *possible* feature space but a few select features are necessary for a particular computation. This is often the case in tree architectures, where each node has an associated feature. The *possible* feature space is all of the features associated with all of the nodes, but the *required* features are only the features associated with nodes that are actually visited. The purpose of [Representations](representation.md) is to allow these features to be calculated as needed instead of calculating all of the features before hand, which is less efficient.
