## [TemplateList](../templatelist/templatelist.md) read() {: #read }

Read all of them templates stored in the [Gallery](gallery.md) from disk into memory. For incremental reads see [readBlock](#readblock)

* **function definition:**

        TemplateList read()

* **parameters:** NONE
* **output:** ([TemplateList](../templatelist/templatelist.md)) Returns a list of all of the templates read from disk
* **example:**

        Gallery *gallery = Gallery::make("gallery_file.xml");
        gallery->read(); // returns a TemplateList of every template stored in the gallery

<!--no italics*-->

## [FileList](../filelist/filelist.md) files() {: #files }

Read all of the filese stored in the [Gallery](gallery.md) from disk into memory.

* **function definition:**

        FileList files()

* **parameters:** NONE
* **output:** ([FileList](../filelist/filelist.md)) Returns a list of all of the files read from disk
* **example:**

        Gallery *gallery = Gallery::make("gallery_file.xml");
        gallery->files(); // returns a FileList of every file stored in the gallery

<!--no italics*-->

## [TemplateList](../templatelist/templatelist.md) readBlock(bool \*done) {: #readblock }

This is a pure virtual function. Incrementally read a block of templates from disk into memory. The size of the block is set by [readBlockSize](properties.md#readblocksize).

* **function definition:**

        virtual TemplateList readBlock(bool *done) = 0

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    done | bool \* | Set to true by the function if the last block has been read, false otherwise

* **output:** ([TemplateList](../templatelist/templatelist.md)) Returns a block of templates loaded from disk
* **example:**

        Gallery *gallery = Gallery::make("gallery_file.xml");

        bool done = false
        while(!done)
            gallery->readBlock(&done); // Each iteration of the loop reads a new block until the end of the gallery is reached

<!--no italics*-->

## void writeBlock(const [TemplateList](../templatelist/templatelist.md) &templates) {: #writeblock }

Write the provided templates to disk. This function calls [write](#write) which should be overloaded by all derived classes. If the gallery is a linked list (see [make](statics.md#make)) each gallery writes the provided templates sequentially.

* **function definition:**

        void writeBlock(const TemplateList &templates)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    templates | const [TemplateList](../templatelist/templatelist.md) & | List of templates to write to disk

* **output:** (void)
* **example:**

        Template t1("picture1.jpg");
        t1.file.set("property", 1);
        Template t2("picture2.jpg");
        t2.file.set("property", 2)

        TemplateList tList = TemplateList() << t1 << t2;

        Gallery *gallery = Gallery::make("gallery_file.xml");
        gallery->writeBlock(tList); // write the templatelist to disk

<!--no italics*-->

## void write(const [Template](../template/template.md) &t) {: #write }

This is a pure virtual function. Write a single template to disk.

* **function definition:**

        virtual void write(const Template &t) = 0

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    t | const [Template](../template/template.md) & | Template to write to disk

* **output:** (void)
* **example:**

        Template t1("picture1.jpg");
        t1.file.set("property", 1);
        Template t2("picture2.jpg");
        t2.file.set("property", 2)

        Gallery *gallery = Gallery::make("gallery_file.xml");
        gallery->write(t1); // write template1 to disk
        gallery->write(t2); // write template2 to disk

## qint64 totalSize() {: #totalsize }

This is a virtual function. Get the total size of the gallery. Default implementation returns <tt>INT_MAX</tt>.

* **function definition:**

        virtual qint64 totalSize()

* **parameters:** NONE
* **output:** (qint64) Returns the total size of the gallery in bytes


## qint64 position() {: #position }

This is a virtual function. Get the current position of the read index in the gallery. The next call to [readBlock](#readblock) will read starting at the reported position.

* **function output:**

        virtual qint64 position()

* **parameters:** NONE
* **output:** (qint64) Returns the current read position in the gallery
* **example:**

        Gallery *gallery = Gallery::make("gallery_file.xml");

        gallery->position(); // returns 0
        bool done; gallery->readBlock(&done);
        gallery->position(); // returns readBlockSize
