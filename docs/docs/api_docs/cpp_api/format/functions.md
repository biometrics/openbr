## [Template](../template/template.md) read() {: #read }

This is a pure virtual function. Read a template from disk at [file](../object/members.md#file).

* **function definition:**

        virtual Template read() const = 0

* **parameters:** NONE
* **output:** ([Template](../template/template.md)) Returns a template loaded from disk
* **example:**

        Format *format = Factory::make<Format>("picture.jpg")
        format->read(); // returns a template loaded from "picture.jpg"

## void write(const [Template](../template/template.md) &t) {: #write }

This is a pure virtual function. Write a provide template to disk at [file](../object/members.md#file)

* **function definition:**

        virtual void write(const Template &t) const = 0

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    t | const [Template](../template/template.md) & | [Template](../template/template.md) to write to disk

* **output:** (void)
* **example:**

        Format *format = Factory::make<Format>("new_pic_location.jpg");

        Template t("picture.jpg");
        format->write(t); // write t to "new_pic_location"
