<!-- TEMPLATE -->

Inherits [QList][QList]&lt;[Mat][Mat]&gt;.

A list of matrices associated with a file.

See:

* [Members](members.md)
* [Constructors](constructors.md)
* [Static Functions](statics.md)
* [Functions](functions.md)

The Template is one of two important data structures in OpenBR (the [File](../file/file.md) is the other).
A template represents a biometric at various stages of enrollment and can be modified by [Transforms](../transform/transform.md) and compared to other [templates](template.md) with [Distances](../distance/distance.md).

While there exist many cases (ex. video enrollment, multiple face detects, per-patch subspace learning, ...) where the template will contain more than one matrix,
in most cases templates have exactly one matrix in their list representing a single image at various stages of enrollment.
In the cases where exactly one image is expected, the template provides the function m() as an idiom for treating it as a single matrix.
Casting operators are also provided to pass the template into image processing functions expecting matrices.

Metadata related to the template that is computed during enrollment (ex. bounding boxes, eye locations, quality metrics, ...) should be assigned to the template's [File](members.md#file) member.

<!-- Links -->
[QList]: http://doc.qt.io/qt-5/QList.html "QList"
[Mat]: http://docs.opencv.org/modules/core/doc/basic_structures.html#mat "Mat"
