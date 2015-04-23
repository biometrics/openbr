# BinaryGallery

An abstract gallery for handling binary data
 

* **file:** gallery/binary.cpp
* **inherits:** [Gallery](../cpp_api/gallery/gallery.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# DefaultGallery

Treats the gallery as a [Format](../cpp_api/format/format.md).
 

* **file:** gallery/default.cpp
* **inherits:** [Gallery](../cpp_api/gallery/gallery.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# EmptyGallery

Reads/writes templates to/from folders.
 

* **file:** gallery/empty.cpp
* **inherits:** [Gallery](../cpp_api/gallery/gallery.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:**

	Property | Type | Description
	--- | --- | ---
	regexp | QString | An optional regular expression to match against the files extension.

---

# FDDBGallery

Implements the FDDB detection format.
 

* **file:** gallery/fddb.cpp
* **inherits:** [Gallery](../cpp_api/gallery/gallery.md)
* **author(s):** [Josh Klontz][jklontz]
* **see:** [http://vis-www.cs.umass.edu/fddb/README.txt](http://vis-www.cs.umass.edu/fddb/README.txt)
* **properties:** None


---

# arffGallery

Weka ARFF file format.
 

* **file:** gallery/arff.cpp
* **inherits:** [Gallery](../cpp_api/gallery/gallery.md)
* **author(s):** [Josh Klontz][jklontz]
* **see:** [http://weka.wikispaces.com/ARFF+%28stable+version%29](http://weka.wikispaces.com/ARFF+%28stable+version%29)
* **properties:** None


---

# aviGallery

Read videos of format .avi
 

* **file:** gallery/video.cpp
* **inherits:** [videoGallery](#videogallery)
* **author(s):** [Unknown][unknown]
* **properties:** None


---

# crawlGallery

Crawl a root location for image files.
 

* **file:** gallery/crawl.cpp
* **inherits:** [Gallery](../cpp_api/gallery/gallery.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# csvGallery

Treats each line as a file.
 

* **file:** gallery/csv.cpp
* **inherits:** [FileGallery](../cpp_api/filegallery/filegallery.md)
* **author(s):** [Josh Klontz][jklontz]
* **format:** Columns should be comma separated with first row containing headers. The first column in the file should be the path to the file to enroll. Other columns will be treated as file metadata. 
* **properties:** None


---

# dbGallery

Database input.
 

* **file:** gallery/db.cpp
* **inherits:** [Gallery](../cpp_api/gallery/gallery.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# flatGallery

Treats each line as a call to [File](../cpp_api/file/file.md)::flat()
 

* **file:** gallery/flat.cpp
* **inherits:** [FileGallery](../cpp_api/filegallery/filegallery.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# galGallery

A binary gallery.

Designed to be a literal translation of templates to disk.
Compatible with [TemplateList](../cpp_api/templatelist/templatelist.md)::fromBuffer.
 

* **file:** gallery/binary.cpp
* **inherits:** [BinaryGallery](#binarygallery)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# googleGallery

Input from a google image search.
 

* **file:** gallery/google.cpp
* **inherits:** [Gallery](../cpp_api/gallery/gallery.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# jsonGallery

Newline-separated JSON objects.
 

* **file:** gallery/binary.cpp
* **inherits:** [BinaryGallery](#binarygallery)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# keyframesGallery

Read key frames of a video with LibAV
 

* **file:** gallery/keyframes.cpp
* **inherits:** [Gallery](../cpp_api/gallery/gallery.md)
* **author(s):** [Ben Klein][bhklein]
* **properties:** None


---

# landmarksGallery

Text format for associating anonymous landmarks with images.
 

* **file:** gallery/landmarks.cpp
* **inherits:** [Gallery](../cpp_api/gallery/gallery.md)
* **author(s):** [Josh Klontz][jklontz]
* **format:** The input should be formatted as follows: <pre><code>file_name:x1,y1,x2,y2,...,xn,yn
file_name:x1,y1,x2,y2,...,xn,yn
...
file_name:x1,y1,x2,y2,...,xn,yn
</code></pre>
* **properties:** None


---

# lmatGallery

Likely matrix format
 

* **file:** gallery/lmat.cpp
* **inherits:** [Gallery](../cpp_api/gallery/gallery.md)
* **author(s):** [Josh Klontz][jklontz]
* **see:** [www.liblikely.org](www.liblikely.org)
* **properties:** None


---

# matrixGallery

Combine all [Template](../cpp_api/template/template.md) into one large matrix and process it as a [Format](../cpp_api/format/format.md)
 

* **file:** gallery/matrix.cpp
* **inherits:** [Gallery](../cpp_api/gallery/gallery.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# memGallery

A gallery held in memory.
 

* **file:** gallery/mem.cpp
* **inherits:** [Gallery](../cpp_api/gallery/gallery.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# mp4Gallery

Read key frames of a .mp4 video file with LibAV
 

* **file:** gallery/keyframes.cpp
* **inherits:** [keyframesGallery](#keyframesgallery)
* **author(s):** [Ben Klein][bhklein]
* **properties:** None


---

# postGallery

Handle POST requests
 

* **file:** gallery/post.cpp
* **inherits:** [Gallery](../cpp_api/gallery/gallery.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# seqGallery

DOCUMENT ME
 

* **file:** gallery/seq.cpp
* **inherits:** [Gallery](../cpp_api/gallery/gallery.md)
* **author(s):** [Unknown][unknown]
* **properties:** None


---

# statGallery

Print [Template](../cpp_api/template/template.md) statistics.
 

* **file:** gallery/stat.cpp
* **inherits:** [Gallery](../cpp_api/gallery/gallery.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# templateGallery

Treat the file as a single binary [Template](../cpp_api/template/template.md).
 

* **file:** gallery/template.cpp
* **inherits:** [Gallery](../cpp_api/gallery/gallery.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# turkGallery

For Amazon Mechanical Turk datasets
 

* **file:** gallery/turk.cpp
* **inherits:** [Gallery](../cpp_api/gallery/gallery.md)
* **author(s):** [Scott Klum][sklum]
* **properties:** None


---

# txtGallery

Treats each line as a file.
 

* **file:** gallery/txt.cpp
* **inherits:** [FileGallery](../cpp_api/filegallery/filegallery.md)
* **author(s):** [Josh Klontz][jklontz]
* **format:** The entire line is treated as the file path. <pre><code>&lt;FILE&gt;
&lt;FILE&gt;
...
&lt;FILE&gt;
</code></pre>An optional label may be specified using a space ' ' separator: <pre><code>&lt;FILE&gt; &lt;LABEL&gt;
&lt;FILE&gt; &lt;LABEL&gt;
...
&lt;FILE&gt; &lt;LABEL&gt;
</code></pre>
* **properties:** None


---

# urlGallery

Newline-separated URLs.
 

* **file:** gallery/binary.cpp
* **inherits:** [BinaryGallery](#binarygallery)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# utGallery

A contiguous array of br_universal_template.
 

* **file:** gallery/binary.cpp
* **inherits:** [BinaryGallery](#binarygallery)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# vbbGallery

DOCUMENT ME
 

* **file:** gallery/vbb.cpp
* **inherits:** [Gallery](../cpp_api/gallery/gallery.md)
* **author(s):** [Unknown][unknown]
* **properties:** None


---

# videoGallery

Read a video frame by frame using cv::VideoCapture
 

* **file:** gallery/video.cpp
* **inherits:** [Gallery](../cpp_api/gallery/gallery.md)
* **author(s):** [Unknown][unknown]
* **properties:** None


---

# webcamGallery

Read a video from the webcam
 

* **file:** gallery/video.cpp
* **inherits:** [videoGallery](#videogallery)
* **author(s):** [Unknown][unknown]
* **properties:** None


---

# wmvGallery

Read videos of format .wmv
 

* **file:** gallery/video.cpp
* **inherits:** [videoGallery](#videogallery)
* **author(s):** [Unknown][unknown]
* **properties:** None


---

# xmlGallery

A sigset input.
 

* **file:** gallery/xml.cpp
* **inherits:** [FileGallery](../cpp_api/filegallery/filegallery.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

