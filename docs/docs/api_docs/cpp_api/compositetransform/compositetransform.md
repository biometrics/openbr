<!-- CompositeTransform -->

Inherits [TimeVaryingTransform](../timevaryingtransform/timevaryingtransform.md)

Base class for [Transforms](../transform/transform.md) that aggregate subtransforms.

See:

* [Properties](properties.md)
* [Members](members.md)
* [Constructors](constructors.md)
* [Static Functions](statics.md)
* [Functions](functions.md)

A CompositeTransform is a wrapper around a list of child transforms. It is used internally for plugins like pipes and forks. It offers