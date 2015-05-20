
## bool trainable() {: #trainable }

This is a virtual function. Check if the distance is trainable. The default version returns true. Distances that are not trainable should derive from [UntrainableDistance](../untrainabledistance/untrainabledistance.md) instead.

* **function defintion:**

        virtual bool trainable()

* **parameters:** NONE
* **output:** (bool) Returns true if the distance is trainable, false otherwise.

## void train(const [TemplateList](../templatelist/templatelist.md) &src) {: #train }

This is a pure virtual function. Train the distance on a provided [TemplateList](../templatelist/templatelist.md) of data. The structure of the data is dependent on the distance to be trained. Distances that are not trainable should derive from [UntrainableDistance](../untrainabledistance/untrainabledistance.md) so they do not have to overload this function.

* **function definition:**

        virtual void train(const TemplateList &src) = 0

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    src | const [TemplateList](../templatelist/templatelist.md) & | Training data for the distance.

* **output:** (void)
* **example:**

        // Create data for a 2-class classification problem
        Template t1("training_pic1.jpg");
        t1.file.set("Label", 0);
        Template t2("training_pic2.jpg");
        t2.file.set("Label", 0);
        Template t3("training_pic3.jpg");
        t3.file.set("Label", 1);
        Template t4("training_pic4.jpg");
        t4.file.set("Label", 1);

        TemplateList training_data(QList<Template>() << t1 << t2 << t3 << t4);

        Transform *distance = Distance::fromAlgorithm("Enrollment:Distance");
        distance->train(training_data); // Images are enrolled through Enrollment and the passed to Distance for training


## void compare(const [TemplateList](../templatelist/templatelist.md) &target, const [TemplateList] &query, [Output](../output/output.md)) {: #compare-1 }

This is a virtual function. Compare two [TemplateLists](../templatelist/templatelist.md) and store the results in a provided output.

* **function definition:**

        virtual void compare(const TemplateList &target, const TemplateList &query, Output *output) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    target | const [TemplateList](../templatelist/templatelist.md) & | List of templates to compare the query against
    query | const [TemplateList](../templatelist/templatelist.md) & | List of templates to compare against the target
    output | [Output](../output/output.md) \* | [Output](../output/output.md) plugin to use to store the results of the comparisons

* **output:** (void)


## [QList][QList]&lt;float&gt; compare(const [TemplateList](../templatelist/templatelist.md) &target, const [Template](../template/template.md) &query) {: #compare-2 }

This is a virtual function. Compare a query against a list of targets. Each comparison results in a floating point response which is the distance between the query and a specific target.

* **function definition:**

        virtual QList<float> compare(const TemplateList &targets, const Template &query) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    targets | const [TemplateList](../templatelist/templatelist.md) & | List of templates to compare the query against
    query | const [Template](../template/template.md) & | Query template to be compared

* **output:** ([QList][QList]&lt;float&gt;) Returns a list of the responses from each comparison between the query and a target.
* **example:**

        Template t1("target_picture1.jpg");
        Template t2("target_picture2.jpg");
        Template t3("target_picture3.jpg");

        TemplateList targets = TemplateList() << t1 << t2 << t3;

        Template query("query_picture.jpg");

        algorithm = "Enrollment:Distance";

        Transform *transform = Transform::fromAlgorithm(algorithm);
        Distance *distance = Distance::fromAlgorithm(algorithm);

        targets >> *transform;
        query   >> *transform;

        distance->compare(targets, query); // returns [0.37, -0.56, 4.35] *Note results are made up!


## float compare(const [Template](../template/template.md) &a, const [Template](../template/template.md) &b) {: #compare-3}

This is a virtual function. Compare two templates and get the difference between them.

* **function definition:**

        virtual float compare(const Template &a, const Template &b) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    a | const [Template](../template/template.md) & | First template to compare
    b | const [Template](../template/template.md) & | Second template to compare

* **output:** (float) Returns the calculated difference between the provided templates
* **example:**

Template a("picture_a.jpg");
Template b("picture_b.jpg");

algorithm = "Enrollment:Distance";

Transform *transform = Transform::fromAlgorithm(algorithm);
Distance *distance = Distance::fromAlgorithm(algorithm);

a >> *transform;
b >> *transform;

distance->compare(a, b); // returns 16.43 *Note results are made up!


## float compare(const [Mat][Mat] &a, const [Mat][Mat] &b) {: #compare-4}

This is a virtual function. Compare two [Mats][Mat] and get the difference between them.

* **function definition:**

        virtual float compare(const Mat &a, const Mat &b) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    a | const [Mat][Mat] & | First matrix to compare
    b | const [Mat][Mat] & | Second matrix to compare

* **output:** (float) Returns the calculated difference between the provided [Mats][Mat]
* **example:**

        Template a("picture_a.jpg");
        Template b("picture_b.jpg");

        algorithm = "Enrollment:Distance";

        Transform *transform = Transform::fromAlgorithm(algorithm);
        Distance *distance = Distance::fromAlgorithm(algorithm);

        a >> *transform;
        b >> *transform;

        distance->compare(a.m(), b.m()); // returns 16.43 *Note results are made up!


## float compare(const uchar \*a, const uchar \*b, size_t size) {: #compare-5 }

This is a virtual function. Compare two buffers and get the difference between them

* **function definition:**

        virtual float compare(const uchar *a, const uchar *b, size_t size) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    a | const uchar \* | First buffer to compare
    b | const uchar \* | Second buffer to compare
    size | size_t | Size of buffers a and b (they must be the same size)

* **output:** (float) Returns the calculated difference between the provided buffers
* **example:**

        Template a("picture_a.jpg");
        Template b("picture_b.jpg");

        algorithm = "Enrollment:Distance";

        Transform *transform = Transform::fromAlgorithm(algorithm);
        Distance *distance = Distance::fromAlgorithm(algorithm);

        a >> *transform;
        b >> *transform;

        distance->compare(a.m().ptr(), b.m().ptr()); // returns -4.32 *Note results are made up!


## [Distance](distance.md) \*make(const [QString][QString] &description) {: #make }

This is a protected function. Makes a child distance from a provided description by calling [make](statics.md#make) with parent = <tt>this</tt>.

* **function definition:**

        inline Distance *make(const QString &description)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    description | const [QString][QString] & | Description of the child distance

* **output:** ([Distance](distance.md) \*) Returns a pointer to the created child distance

<!-- Links -->
[QString]: http://doc.qt.io/qt-5/QString.html "QString"
[QList]: http://doc.qt.io/qt-5/QList.html "QList"
[Mat]: http://docs.opencv.org/modules/core/doc/basic_structures.html#mat "Mat"
