## [QDataStream][QDataStream] &operator<<([QDataStream][QDataStream] &stream, const [Template](template.md) &t) {: #operator-ltlt }

Serialize a template

* **function definition:**

        QDataStream &operator<<(QDataStream &stream, const Template &t)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    stream | [QDataStream][QDataStream] & | The stream to serialize to
    t | const [Template](template.md) & | The template to serialize

* **output:** ([QDataStream][QDataStream] &) Returns the updated stream
* **example:**

        void store(QDataStream &stream)
        {
            Template t("picture.jpg");
            t.append(Mat::ones(1, 1, CV_8U));

            stream << t; // "["1"]picture.jpg" serialized to the stream
        }

## [QDataStream][QDataStream] &operator>>([QDataStream][QDataStream] &stream, [Template](template.md) &t) {: #operator-gtgt }

Deserialize a template

* **function definition:**

        QDataStream &operator>>(QDataStream &stream, Template &t)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    stream | [QDataStream][QDataStream] & | The stream to deserialize to
    t | const [Template](template.md) & | The template to deserialize

* **output:** ([QDataStream][QDataStream] &) Returns the updated stream
* **example:**

        void load(QDataStream &stream)
        {
            Template in("picture.jpg");
            in.append(Mat::ones(1, 1, CV_8U));

            stream << in; // "["1"]picture.jpg" serialized to the stream

            Template out;
            stream >> out;

            out.file; // returns "picture.jpg"
            out; // returns ["1"]
        }

<!-- Links -->
[QDataStream]: http://doc.qt.io/qt-5/qdatastream.html "QDataStream"
