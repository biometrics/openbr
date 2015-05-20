## operator const [File](../file/file.md) &() const {: #operator-file }

Idiom to treat the template like a [File](../file/file.md).

* **function definition:**

        inline operator const File &() const

* **parameters:** NONE
* **output:** ([File](../file/file.md) Returns [file](members.md#file).


## const [Mat][Mat] &m() const {: #m-1 }

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


## [Mat][Mat] &m() {: #m-2 }

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


## operator const [Mat][Mat] &() {: #operator-mat-1 }

Idiom to treat the template like a [Mat][Mat]. Makes a call to [m()](#m-1).

* **function definition:**

        inline operator const Mat&() const

* **parameters:** NONE
* **output:** ([Mat][Mat]) Returns the last [Mat][Mat] in the list. If the list is empty an empty [Mat][Mat] is returned.
* **see:** [m](#m-1)


## operator [Mat][Mat] &() {: #operator-mat-2 }

Idiom to treat the template like a [Mat][Mat]. Makes a call to [m()](#m-1).

* **function definition:**

        inline operator Mat&()

* **parameters:** NONE
* **output:** ([Mat][Mat]) Returns the last [Mat][Mat] in the list. If the list is empty an empty [Mat][Mat] is returned.
* **see:** [m](#m-1)


## operator [_InputArray][InputArray] &() {: #operator-inputarray }

Idiom to treat the template like an [_InputArray][InputArray]. Makes a call to [m()](#m-1).

* **function definition:**

        inline operator _InputArray() const

<!-- _no italics_-->
* **parameters:** NONE
* **output:** ([_InputArray][InputArray]) Returns the last [Mat][Mat] in the list. If the list is empty an empty [Mat][Mat] is returned.
* **see:** [m](#m-1)


## operator [_OutputArray][OutputArray] &() {: #operator-outputarray }

Idiom to treat the template like an [_OutputArray][InputArray]. Makes a call to [m()](#m-1).

* **function definition:**

        inline operator _OutputArray()

* **parameters:** NONE
* **output:** ([_OutputArray][OutputArray]) Returns the last [Mat][Mat] in the list. If the list is empty an empty [Mat][Mat] is returned.
* **see:** [m](#m-1)


## [Mat][Mat] &operator =(const [Mat][Mat] &other) {: #operator-e }

Idiom to treat the template like a [Mat][Mat]. Set the result of [m()](#m-1) equal to other.

* **function definition:**

        inline Mat &operator=(const Mat &other)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    other | const Mat & | Mat to overwrite value of [m](#m-1)

* **output**: ([Mat][Mat] &) Returns a reference to the updated [Mat][Mat]


## bool isNull() const {: #isnull }

Check if the template is NULL.

* **function definition:**

        inline bool isNull() const

* **parameters:** NONE
* **output:** (bool) Returns true if the template is empty *or* if [m](#m-1) has no data.
* **example:**

        Template t;
        t.isNull(); // returns true

        t.append(Mat());
        t.isNull(); // returns true

        t.append(Mat::ones(1, 1, CV_8U));
        t.isNull(); // returns false


## void merge(const [Template](template.md) &other) {: #merge }

Append the contents of another template. The [files](members.md#file) are appended using [append](../file/functions.md#append-1).

* **function definition:**

        inline void merge(const Template &other)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    other | const [Template](template.md) & | Template to be merged

* **output:** (void)
* **example:**

        Template t1("picture1.jpg"), t2("picture2.jpg");
        Mat m1, m2;

        t1.append(m1);
        t2.append(m2);

        t1.merge(t2);

        t1.file; // returns picture1.jpg;picture2.jpg[seperator=;]
        t1; // returns [m1, m2]


## size_t bytes() const {: #bytes }

Get the total number of bytes in the template

* **function definition:**

        size_t bytes() const

* **parameters:** None
* **output:** (int) Returns the sum of the bytes in each [Mat][Mat] in the [Template](template.md)
* **example:**

        Template t;

        Mat m1 = Mat::ones(1, 1, CV_8U); // 1 byte
        Mat m2 = Mat::ones(2, 2, CV_8U); // 4 bytes
        Mat m3 = Mat::ones(3, 3, CV_8U); // 9 bytes

        t << m1 << m2 << m3;

        t.bytes(); // returns 14


## Template clone() const {: #clone }

Clone the template

* **function definition:**

        Template clone() const

* **parameters:** NONE
* **output:** ([Template](template.md)) Returns a new [Template](template.md) with copies of the [file](members.md#file) and each [Mat][Mat] that was in the original.
* **example:**

        Template t1("picture.jpg");
        t1.append(Mat::ones(1, 1, CV_8U));

        Template t2 = t1.clone();

        t2.file; // returns "picture.jpg"
        t2; // returns ["1"]

<!-- Links -->
[Mat]: http://docs.opencv.org/modules/core/doc/basic_structures.html#mat "Mat"
[InputArray]: http://docs.opencv.org/modules/core/doc/basic_structures.html#inputarray "InputArray"
[OutputArray]: http://docs.opencv.org/modules/core/doc/basic_structures.html#outputarray "OutputArray"
