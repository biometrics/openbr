## void projectMetadata(const [File](../file/file.md) &src, [File](../file/file.md) &dst) {: #projectmetadata }

This is a pure virtual function. It must be overloaded by all derived classes. Project a [Template's](../template/template.md) [metadata](../template/members.md#file) through the transform, modifying its contents in some way and storing the modified data in **dst**.

* **function definition:**

        virtual void projectMetadata(const File &src, File &dst) const = 0

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    src | const [File](../file/file.md) & | Input File. It is immutable
    dst | [File](../file/file.md) & | Output File. Should contain the modified data from the input template.

* **output:** (void)
* **example:**

        class IncrementPropertyTransform : public MetadataTransform
        {
            Q_OBJECT
            Q_PROPERTY(QString key READ get_key WRITE set_key RESET reset_key STORED false)
            BR_PROPERTY(QString, key, "")

            void projectMetadata(const File &src, File &dst) const
            {
                dst = src;
                dst.set(key, src.get<int>(key, 0) + 1);
            }
        };

        BR_REGISTER(Transform, IncrementPropertyTransform)

        MetadataTransform *m_transform = (MetadataTransform *)Transform::make("IncrementProperty(property1)", NULL);

        File in("picture.jpg"), out;
        in.set("property1", 10);

        m_transform->projectMetadata(in, out);
        out.flat(); // Returns "picture.jpg[property1=11]"


## void project(const [Template](../template/template.md) &src, [Template](../template/template.md) &dst) {: #project }

Project a [Template](../template/template.md) through the transform by passing its [metadata](../template/members.md#file) through [projectMetadata](#projectmetadata) and storing the result in dst. All matrices in src are passed unchanged to dst.

* **function definition:**

        void project(const Template &src, Template &dst) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    src | const [Template](../template/template.md) & | Input Template. It is immutable
    dst | [Template](../template/template.md) & | Output Template. Should contain the modified data from the input template.

* **output:** (void)
