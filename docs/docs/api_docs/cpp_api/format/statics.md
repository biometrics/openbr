## [Template](../template/template.md) read(const [QString][QString] &file) {: #read }

Read a [Template](../template/template.md) from disk at the provide file location. A derived class format is chosen based on the suffix of the provided file.

* **function definition:**

static Template read(const QString &file)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    file | const [QString][QString] & | File to load a [Template](../template/template.md) from.

* **output:** ([Template](../template/template.md)) Returns a [Template](../template/template.md) loaded from disk.
* **example:**

        Format::read("picture.jpg"); // returns a template loaded from "picture.jpg". The proper Format to load jpg images is selected automatically


## void write(const [QString][QString] &file, const [Template](../template/template.md) &t) {: #write }

Write a template to disk at the provided file location.

* **function definition:**

        static void write(const QString &file, const Template &t)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    file | const [QString][QString] & | File to write a [Template](../template/template.md) to
    t | const [Template](../template/template.md) & | [Template](../template/template.md) to write to disk

* **output:** (void)
* **example:**

        Template t("picture.jpg");

        Format::write("new_pic_location.jpg", t); // Write t to "new_pic_location.jpg"

<!-- Links -->
[QString]: http://doc.qt.io/qt-5/QString.html "QString"
