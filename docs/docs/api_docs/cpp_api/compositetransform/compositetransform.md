<!-- CompositeTransform -->

Inherits [TimeVaryingTransform](../timevaryingtransform/timevaryingtransform.md)

Base class for [Transforms](../transform/transform.md) that aggregate subtransforms.

See:

* [Properties](properties.md)
* [Members](members.md)
* [Constructors](constructors.md)
* [Functions](functions.md)

CompositeTransforms are the base class for [Transforms](../transform/transform.md) that have a list of child transforms. It is used internally for plugins like pipes and forks. It inherits from [TimeVaryingTransform](../timevaryingtransform/timevaryingtransform.md) so that it can properly handle having children (one or many) that are time varying.
