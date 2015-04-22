## [FileList](filelist.md) fromGallery(const [File](../file/file.md) &gallery, bool cache = false) {: #fromgallery }

Create a [FileList](filelist.md) from a [Gallery](../gallery/gallery.md). Galleries store one or more [Templates](../template/template.md) on disk. Common formats include **csv**, **xml**, and **gal**, which is a unique OpenBR format. Read more about this in the [Gallery](../gallery/gallery.md) section. This function creates a [FileList](filelist.md) by parsing the stored gallery based on its format. Cache determines whether the gallery should be stored for faster reading later.

* **function definition:**

        static FileList fromGallery(const File &gallery, bool cache = false)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    gallery | const [File](../file/file.md) & | Gallery file to be enrolled
    cache | bool | (Optional) Retain the gallery in memory. Default is false.

* **output:** ([FileList](filelist.md)) Returns the filelist that the gallery was enrolled into
* **example:**

        File gallery("gallery.csv");

        FileList fList = FileList::fromGallery(gallery);
        fList.flat(); // returns all the files that have been loaded from disk. It could
                      // be 1 or 100 depending on what was stored.
