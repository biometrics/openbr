<!-- GALLERY -->

Inherits [Object](../object/object.md)

Plugin base class for storing a list of enrolled templates.

See:

* [Properties](properties.md)
* [Constructors](constructors.md)
* [Static Functions](statics.md)
* [Functions](functions.md)

A *gallery* is a file representing a [TemplateList](../templatelist/templatelist.md) serialized to disk. [File](../file/file.md)::[suffix](../file/functions.md#suffix) is used to determine which plugin should handle the gallery. The currently supported extensions are

* xml
* avi
* wmv
* mp4
* webcam
* vbb (OpenCV format)
* txt
* turk
* template
* stat
* seq
* post
* mem (**NOTE:** Mem galleries live only in RAM; they should be used for caching and not for normal I/O)
* matrix
* landmarks
* google
* flat
* FDDB
* db
* csv
* crawl
* gal
* ut
* url
* json
* arff

Many of these extensions are unique to OpenBR. Please look at the relevant [Gallery plugin](../../../plugin_docs/gallery.md) for information on formatting and other concerns.
