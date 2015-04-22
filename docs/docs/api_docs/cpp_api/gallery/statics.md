## [Gallery](gallery.md) \*make(const [File](../file/file.md) &file) {: #make }

Make a [Gallery](gallery.md) from a string. The provided file is first split using [File](../file/file.md)::[split](../file/functions.md#split-1) and each resulting file is turned into a [Gallery](gallery.md) that is stored in a linked-list.

* **function definition:**

        static Gallery *make(const File &file)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    file | const [File](../file/file.md) & | File describing the gallery or galleries to construct

* **output:** ([Gallery](gallery.md) \*) Returns a pointer to the first gallery in the linked list
* **example:**

        Gallery *gallery1 = Gallery::make("gallery_file.xml"); // returns a pointer to the gallery
        Gallery *gallery2 = Gallery::make("gallery_file1.xml;gallery_file2.xml"); // returns a pointer to the gallery created with "gallery_file1.xml" with a pointer to the gallery created with "gallery_file2.xml"
