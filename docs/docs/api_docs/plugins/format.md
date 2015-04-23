# DefaultFormat

Reads image files.
 

* **file:** format/video.cpp
* **inherits:** [Format](../cpp_api/format/format.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# binaryFormat

A simple binary matrix format.
 

* **file:** format/binary.cpp
* **inherits:** [Format](../cpp_api/format/format.md)
* **author(s):** [Josh Klontz][jklontz]
* **format:** First 4 bytes indicate the number of rows. Second 4 bytes indicate the number of columns. The rest of the bytes are 32-bit floating data elements in row-major order. 
* **properties:** None


---

# csvFormat

Reads a comma separated value file.
 

* **file:** format/csv.cpp
* **inherits:** [Format](../cpp_api/format/format.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# ebtsFormat

Reads FBI EBTS transactions.
 

* **file:** format/ebts.cpp
* **inherits:** [Format](../cpp_api/format/format.md)
* **author(s):** [Scott Klum][sklum]
* **see:** [https://www.fbibiospecs.org/ebts.html](https://www.fbibiospecs.org/ebts.html)
* **properties:** None


---

# lffsFormat

Reads a NIST LFFS file.
 

* **file:** format/lffs.cpp
* **inherits:** [Format](../cpp_api/format/format.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# lmatFormat

Likely matrix format
 

* **file:** format/lmat.cpp
* **inherits:** [Format](../cpp_api/format/format.md)
* **author(s):** [Josh Klontz][jklontz]
* **see:** [www.liblikely.org](www.liblikely.org)
* **properties:** None


---

# maskFormat

Reads a NIST BEE mask matrix.
 

* **file:** format/mtx.cpp
* **inherits:** [mtxFormat](#mtxformat)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# matFormat

MATLAB <tt>.mat</tt> format.

matFormat is known not to work with compressed matrices
 

* **file:** format/mat.cpp
* **inherits:** [Format](../cpp_api/format/format.md)
* **author(s):** [Josh Klontz][jklontz]
* **see:** [http://www.mathworks.com/help/pdf_doc/matlab/matfile_format.pdf](http://www.mathworks.com/help/pdf_doc/matlab/matfile_format.pdf)
* **properties:** None


---

# mtxFormat

Reads a NIST BEE similarity matrix.
 

* **file:** format/mtx.cpp
* **inherits:** [Format](../cpp_api/format/format.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# nullFormat

Returns an empty matrix.
 

* **file:** format/null.cpp
* **inherits:** [Format](../cpp_api/format/format.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# postFormat

Handle POST requests
 

* **file:** format/post.cpp
* **inherits:** [Format](../cpp_api/format/format.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# rawFormat

RAW format
 

* **file:** format/raw.cpp
* **inherits:** [Format](../cpp_api/format/format.md)
* **author(s):** [Josh Klontz][jklontz]
* **see:** [http://www.nist.gov/srd/nistsd27.cfm](http://www.nist.gov/srd/nistsd27.cfm)
* **properties:** None


---

# scoresFormat

Reads in scores or ground truth from a text table.
 

* **file:** format/scores.cpp
* **inherits:** [Format](../cpp_api/format/format.md)
* **author(s):** [Josh Klontz][jklontz]
* **format:** Example of the format: <pre><code>2.2531514    FALSE   99990377    99990164
2.2549822    TRUE    99990101    99990101
</code></pre>
* **properties:** None


---

# urlFormat

Reads image files from the web.
 

* **file:** format/url.cpp
* **inherits:** [Format](../cpp_api/format/format.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# videoFormat

Read all frames of a video using OpenCV
 

* **file:** format/video.cpp
* **inherits:** [Format](../cpp_api/format/format.md)
* **author(s):** [Charles Otto][caotto]
* **properties:** None


---

# webcamFormat

Retrieves an image from a webcam.
 

* **file:** format/video.cpp
* **inherits:** [Format](../cpp_api/format/format.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# xmlFormat

Decodes images from Base64 xml
 

* **file:** format/xml.cpp
* **inherits:** [Format](../cpp_api/format/format.md)
* **author(s):** [Scott Klum][sklum], [Josh Klontz][jklontz]
* **properties:** None


---

