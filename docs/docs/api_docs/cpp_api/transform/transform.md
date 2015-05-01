<!-- TRANSFORM -->

Inherits [Object](../object/object.md)

Plugin base class for processing a template.

See:

* [Members](members.md)
* [Constructors](constructors.md)
* [Static Functions](statics.md)
* [Functions](functions.md)

Transforms support the idea of *training* and *projecting*, whereby they are (optionally) given example images and are expected learn how to transform new instances into an alternative, hopefully more useful, basis for the recognition task at hand. Transforms can be chained together to support the declaration and use of arbitrary algorithms at run time.
