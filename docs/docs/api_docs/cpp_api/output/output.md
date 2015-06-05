<!-- OUTPUT -->

Inherits from [Object](../object/object.md)

Plugin base class for storing template comparison results.

See:

* [Members](members.md)
* [Constructors](constructors.md)
* [Static Functions](statics.md)
* [Functions](functions.md)

An *Output* is a [File](../file/file.md) representing the result comparing templates. [File](../file/file.md)::[suffix](../file/functions.md#suffix) is used to determine which plugin should handle the output. The currently supported extensions are:

* txt
* tail
* rr
* rank
* null
* mtx
* melt
* hist
* heat
* eval
* csv
* best

Many of these extensions are unique to OpenBR. Please look at the relevant [Output plugin](../../../plugin_docs/output.md) for information on formatting and other concerns.
