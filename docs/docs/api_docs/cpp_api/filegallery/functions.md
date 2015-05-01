## void init() {: #init }

Initialize the [FileGallery](filegallery.md). This sets [f](members.md#f) using the file name from [file](../object/members.md#file). It also calls [Gallery](../gallery/gallery.md)::[init](../object/functions.md#init).

* **function definition:**

        void init()

* **parameters:** NONE
* **output:** (void)

## qint64 totalSize() {: #totalsize }

Get the total size of the file. This is useful for estimating progress.

* **function definition:**

        qint64 totalSize()

* **parameters:** NONE
* **output:** (qint64) Returns the total size of the file in bytes

## qint64 position() {: #pos }

Get the current index in the file. This is useful for reading and writing blocks of data

* **function definition:**

        qint64 position()

* **parameters:** NONE
* **output:** (qint64) Returns the current position in the file

## bool readOpen() {: #readopen }

Open [f](members.md#f) in read-only mode

* **function definition:**

        bool readOpen()

* **parameters:** NONE
* **output:** (bool) Returns true if the file was opened successfully, false otherwise

## void writeOnly() {: #writeonly }

Open [f](members.md#f) in write-only mode

* **function definition:**

        void writeOpen()

* **parameters:** NONE
* **output:** (void)
