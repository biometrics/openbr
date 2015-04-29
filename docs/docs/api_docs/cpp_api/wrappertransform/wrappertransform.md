<!-- WrapperTransform -->

Inherits [TimeVaryingTransform](../timevaryingtransform/timevaryingtransform.md)

Base class for [Transforms](../transform/transform.md) that have a single child transform

See:

* [Properties](properties.md)
* [Constructors](constructors.md)
* [Functions](functions.md)

WrapperTransforms are the base class for plugins that have a child transform. It inherits from [TimeVaryingTransform](../timevaryingtransform/timevaryingtransform.md) so that it can properly handle a child transform that is also time varying, WrapperTransform itself has no requirement to be time varying. The main purpose of WrapperTransform is to intelligently implement [simplify](functions.md#simplify) and [smartCopy](functions.md#smartcopy), all other calls are just passed to the child.
