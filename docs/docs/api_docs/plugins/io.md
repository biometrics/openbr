# DecodeTransform

Decodes images
 

* **file:** io/decode.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# DownloadTransform

Downloads an image from a URL
 

* **file:** io/download.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# GalleryOutputTransform

DOCUMENT ME
 

* **file:** io/galleryoutput.cpp
* **inherits:** [TimeVaryingTransform](../cpp_api/timevaryingtransform/timevaryingtransform.md)
* **author(s):** [Unknown][unknown]
* **properties:** None


---

# IncrementalOutputTransform

Incrementally output templates received to a gallery, based on the current filename

When a template is received in projectUpdate for the first time since a finalize, open a new gallery based on the
template's filename, and the galleryFormat property.

[Template](../cpp_api/template/template.md) received in projectUpdate will be output to the gallery with a filename combining their original filename and
their FrameNumber property, with the file extension specified by the fileFormat property.
 

* **file:** io/incrementaloutput.cpp
* **inherits:** [TimeVaryingTransform](../cpp_api/timevaryingtransform/timevaryingtransform.md)
* **author(s):** [Charles Otto][caotto]
* **properties:** None


---

# OpenTransform

Applies [Format](../cpp_api/format/format.md) to [Template](../cpp_api/template/template.md) filename and appends results.
 

* **file:** io/open.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# OutputTransform

DOCUMENT ME
 

* **file:** io/out.cpp
* **inherits:** [TimeVaryingTransform](../cpp_api/timevaryingtransform/timevaryingtransform.md)
* **author(s):** [Unknown][Unknown]
* **properties:** None


---

# PrintTransform

Prints the file of the input [Template](../cpp_api/template/template.md) to stdout or stderr.
 

* **file:** io/print.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# ReadLandmarksTransform

Read landmarks from a file and associate them with the correct [Template](../cpp_api/template/template.md).
 

* **file:** io/readlandmarks.cpp
* **inherits:** [UntrainableMetadataTransform](../cpp_api/untrainablemetadatatransform/untrainablemetadatatransform.md)
* **author(s):** [Scott Klum][sklum]
* **format:** Example of the format: <pre><code>image_001.jpg:146.000000,190.000000,227.000000,186.000000,202.000000,256.000000
image_002.jpg:75.000000,235.000000,140.000000,225.000000,91.000000,300.000000
image_003.jpg:158.000000,186.000000,246.000000,188.000000,208.000000,233.000000
</code></pre>
* **properties:** None


---

# ReadTransform

Read images
 

* **file:** io/read.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# WriteTransform

Write all mats to disk as images.
 

* **file:** io/write.cpp
* **inherits:** [TimeVaryingTransform](../cpp_api/timevaryingtransform/timevaryingtransform.md)
* **author(s):** [Brendan Klare][bklare]
* **properties:** None


---

# YouTubeFacesDBTransform

Implements the YouTubesFaceDB
 

* **file:** io/youtubefacesdb.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **read:**

	1. *Wolf, Lior, Tal Hassner, and Itay Maoz.*
	 **"Face recognition in unconstrained videos with matched background similarity."**
	 Computer Vision and Pattern Recognition (CVPR), 2011 IEEE Conference on. IEEE, 2011.

* **properties:** None


---

