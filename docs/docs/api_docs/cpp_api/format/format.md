<!-- FORMAT -->

Inherits [Object](../object/object.md)

Plugin base class for reading a template from disk.

See:

* [Constructors](constructors.md)
* [Static Functions](statics.md)
* [Functions](functions.md)

A *format* is a [File](../file/file.md) representing a [Template](../template/template.md) on disk. [File](../file/file.md)::[suffix](../file/functions.md#suffix) is used to determine which derived format plugin should handle a file. Currently supported extensions are:

* [OpenCV image formats][OpenCV Image Formats]
* xml
* scores
* url
* raw
* post
* null
* mtx
* mask
* mat
* lffs
* ebts
* csv
* binary

Many of these extensions are unique to OpenBR. Please look at the relevant [Format plugin](../../../plugin_docs/format.md) for information on formatting and other concerns.

<!-- Links -->
[OpenCV Image Formats]: http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=imread#imread "OpenCV Image Formats"
