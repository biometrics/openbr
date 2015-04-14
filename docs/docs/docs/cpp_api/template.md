<!-- TEMPLATE -->

Inherits [QList][QList]&lt;[Mat][Mat]&gt;.

A list of matrices associated with a file.

The Template is one of two important data structures in OpenBR (the [File](#file) is the other).
A template represents a biometric at various stages of enrollment and can be modified by [Transforms](#transform) and compared to other [templates](#template) with [Distance](#distance).

While there exist many cases (ex. video enrollment, multiple face detects, per-patch subspace learning, ...) where the template will contain more than one matrix,
in most cases templates have exactly one matrix in their list representing a single image at various stages of enrollment.
In the cases where exactly one image is expected, the template provides the function m() as an idiom for treating it as a single matrix.
Casting operators are also provided to pass the template into image processing functions expecting matrices.

Metadata related to the template that is computed during enrollment (ex. bounding boxes, eye locations, quality metrics, ...) should be assigned to the template's [File](#template-members-file) member.

## Members {: #template-members }

Member | Type | Description
--- | --- | ---
<a class="table-anchor" id=template-members-file></a>file | [File](#file) | The file that constructs the template and stores its associated metadata

---

## Constructors {: #template-constructors }

Constructor | Description
--- | ---
Template() | The default template constructor. It doesn't do anything.
Template(const [File](#file) &file) | Sets [file](#template-members-file) to the given [File](#file).
Template(const [File](#file) &file, const [Mat][Mat] &mat) | Sets [file](#template-members-file) to the given [File](#file) and appends the given [Mat][Mat] to itself.
Template(const [Mat][Mat] &mat) | Appends the given [Mat][Mat] to itself

---

## Static Functions {: #template-static-functions }


### [QDataStream][QDataStream] &operator<<([QDataStream][QDataStream] &stream, const [Template](#template) &t) {: #template-static-operator-ltlt }

Serialize a template

* **function definition:**

        QDataStream &operator<<(QDataStream &stream, const Template &t)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    stream | [QDataStream][QDataStream] & | The stream to serialize to
    t | const [Template](#template) & | The template to serialize

* **output:** ([QDataStream][QDataStream] &) Returns the updated stream
* **example:**

        void store(QDataStream &stream)
        {
            Template t("picture.jpg");
            t.append(Mat::ones(1, 1, CV_8U));

            stream << t; // "["1"]picture.jpg" serialized to the stream
        }

### [QDataStream][QDataStream] &operator>>([QDataStream][QDataStream] &stream, [Template](#template) &t) {: #template-static-operator-gtgt }

Deserialize a template

* **function definition:**

        QDataStream &operator>>(QDataStream &stream, Template &t)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    stream | [QDataStream][QDataStream] & | The stream to deserialize to
    t | const [Template](#template) & | The template to deserialize

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

---

## Functions {: #template-functions }


### operator const [File](#file) &() const {: #template-function-operator-file }

Idiom to treat the template like a [File](#file).

* **function definition:**

        inline operator const File &() const

* **parameters:** NONE
* **output:** ([File](#file) Returns [file](#template-members-file).


### const [Mat][Mat] &m() const {: #template-function-m-1 }

Idiom to treat the template like a [Mat][Mat].

* **function definition:**

        inline const Mat &m() const

* **parameters:** NONE
* **output:** ([Mat][Mat]) Returns the last [Mat][Mat] in the list. If the list is empty an empty [Mat][Mat] is returned.
* **example:**

        Template t;
        t.m(); // returns empty mat

        Mat m1;
        t.append(m1);
        t.m(); // returns m1;

        Mat m2;
        t.append(m2);
        t.m(); // returns m2;


### [Mat][Mat] &m() {: #template-function-m-2 }

Idiom to treat the template like a [Mat][Mat].

* **function definition:**

        inline Mat &m()

* **parameters:** NONE
* **output:** ([Mat][Mat]) Returns the last [Mat][Mat] in the list. If the list is empty an empty [Mat][Mat] is returned.
* **example:**

        Template t;
        t.m(); // returns empty mat

        Mat m1;
        t.append(m1);
        t.m(); // returns m1;

        Mat m2;
        t.append(m2);
        t.m(); // returns m2;


### operator const [Mat][Mat] &() {: #template-function-operator-mat-1 }

Idiom to treat the template like a [Mat][Mat]. Makes a call to [m()](#template-function-m-1).

* **function definition:**

        inline operator const Mat&() const

* **parameters:** NONE
* **output:** ([Mat][Mat]) Returns the last [Mat][Mat] in the list. If the list is empty an empty [Mat][Mat] is returned.
* **see:** [m](#template-function-m-1)


### operator [Mat][Mat] &() {: #template-function-operator-mat-2 }

Idiom to treat the template like a [Mat][Mat]. Makes a call to [m()](#template-function-m-1).

* **function definition:**

        inline operator Mat&()

* **parameters:** NONE
* **output:** ([Mat][Mat]) Returns the last [Mat][Mat] in the list. If the list is empty an empty [Mat][Mat] is returned.
* **see:** [m](#template-function-m-1)


### operator [_InputArray][InputArray] &() {: #template-function-operator-inputarray }

Idiom to treat the template like an [_InputArray][InputArray]. Makes a call to [m()](#template-function-m-1).

* **function definition:**

        inline operator _InputArray() const

<!-- _no italics_-->
* **parameters:** NONE
* **output:** ([_InputArray][InputArray]) Returns the last [Mat][Mat] in the list. If the list is empty an empty [Mat][Mat] is returned.
* **see:** [m](#template-function-m-1)


### operator [_OutputArray][OutputArray] &() {: #template-function-operator-outputarray }

Idiom to treat the template like an [_OutputArray][InputArray]. Makes a call to [m()](#template-function-m-1).

* **function definition:**

        inline operator _OutputArray()

<!-- _no italics_-->
* **parameters:** NONE
* **output:** ([_OutputArray][OutputArray]) Returns the last [Mat][Mat] in the list. If the list is empty an empty [Mat][Mat] is returned.
* **see:** [m](#template-function-m-1)


### [Mat][Mat] &operator =(const [Mat][Mat] &other) {: #template-function-operator-e }

Idiom to treat the template like a [Mat][Mat]. Set the result of [m()](#template-function-m-1) equal to other.

* **function definition:**

        inline Mat &operator=(const Mat &other)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    other | const Mat & | Mat to overwrite value of [m](#template-function-m-1)

* **output**: ([Mat][Mat] &) Returns a reference to the updated [Mat][Mat]


### bool isNull() const {: #template-function-isnull }

Check if the template is NULL.

* **function definition:**

        inline bool isNull() const

* **parameters:** NONE
* **output:** (bool) Returns true if the template is empty *or* if [m](#template-function-m-1) has no data.
* **example:**

        Template t;
        t.isNull(); // returns true

        t.append(Mat());
        t.isNull(); // returns true

        t.append(Mat::ones(1, 1, CV_8U));
        t.isNull(); // returns false


### void merge(const [Template](#template) &other) {: #template-function-merge }

Append the contents of another template. The [files](#template-members-file) are appended using [append](#file-function-append-1).

* **function definition:**

        inline void merge(const Template &other)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    other | const [Template][#template] & | Template to be merged

* **output:** (void)
* **example:**

        Template t1("picture1.jpg"), t2("picture2.jpg");
        Mat m1, m2;

        t1.append(m1);
        t2.append(m2);

        t1.merge(t2);

        t1.file; // returns picture1.jpg;picture2.jpg[seperator=;]
        t1; // returns [m1, m2]


### size_t bytes() const {: #template-function-bytes }

Get the total number of bytes in the template

* **function definition:**

        size_t bytes() const

* **parameters:** None
* **output:** (int) Returns the sum of the bytes in each [Mat][Mat] in the [Template](#template)
* **example:**

        Template t;

        Mat m1 = Mat::ones(1, 1, CV_8U); // 1 byte
        Mat m2 = Mat::ones(2, 2, CV_8U); // 4 bytes
        Mat m3 = Mat::ones(3, 3, CV_8U); // 9 bytes

        t << m1 << m2 << m3;

        t.bytes(); // returns 14


### Template clone() const {: #template-function-clone }

Clone the template

* **function definition:**

        Template clone() const

* **parameters:** NONE
* **output:** ([Template](#template)) Returns a new [Template](#template) with copies of the [file](#template-members-file) and each [Mat][Mat] that was in the original.
* **example:**

        Template t1("picture.jpg");
        t1.append(Mat::ones(1, 1, CV_8U));

        Template t2 = t1.clone();

        t2.file; // returns "picture.jpg"
        t2; // returns ["1"]
