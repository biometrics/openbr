## [Transform](transform.md) \*clone() {: #clone }

This is a virtual function. Make a deep copy of the transform.

* **function definition:**

        virtual Transform *clone() const

* **parameters:** NONE
* **output:** ([Transform](transform.md))
* **example:**

        class ExampleTransform : public Transform
        {
            Q_OBJECT

            Q_PROPERTY(int property READ get_property WRITE set_property RESET reset_property STORED false)
            BR_PROPERTY(int, property, 1)

            ...
        };

        Transform *transform = Transform::make("Example", NULL);
        transform->parameters(); // returns ["property = 1"]

        Transform *clone = transform->clone();
        clone->parameters(); // returns ["property = 1"]

        transform->setProperty("property", 10);
        transform->parameters(); // returns ["property = 10"]
        clone->parameters(); // returns ["property = 1"]


## void train(const [TemplateList](../templatelist/templatelist.md) &data) {: #train-1 }

This is a virtual function. Train the transform on provided training data. This function should be overloaded for any transform that needs to be trained. [Trainable](members.md#trainable) must be set to true for this function to be called. If [independent](members.md#independent) is true a new instance of the transform will be trained for each [Mat][Mat] stored in a [Template](../template/template.md) in the provided training data. Each [Template](../template/template.md) should have the same number of [Mats][Mat].

* **function definition:**

        virtual void train(const TemplateList &data)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    data | const [TemplateList](../templatelist/templatelist.md) & | Training data. The format of the data depends on the transform to be trained. In some cases the transform requires a "Label" field in each [Template](../template/template.md) [file's](../template/members.md#file) [metadata](../file/members.md#m_metadata) (normally these are classifiers like [SVM](../../../plugin_docs/classification.md#svmtransform)). In other cases no metadata is required and training occurs on the raw image data only.

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

        Transform *classifier = Transform::make("DataPrep+Classifier");
        classifier->train(training_data); // The data is projected through DataPrep, assuming it is untrainable, and then passed to the train method of Classifier


## void train(const [QList][QList]&lt;[TemplateList](../templatelist/templatelist.md)&gt; &data) {: #train-2 }

This is a virtual function. This version of train is meant for special-case, internal, transforms and for tranforms that require a specific number of templates at project time. If a transform requires a specific number of templates at project time it should be trained in batches that have the same number of templates. For example, if a transform requires exactly 5 templates when projecting it should get a list of [TemplateLists](../templatelist/templatelist.md), each with exactly 5 templates, at train time. Each [TemplateList](../templatelist/templatelist.md) can then be treated as an individual input to the training function.

* **function definition:**

    virtual void train(const QList<TemplateList> &data)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    data | const [QList][QList]&lt;[TemplateList](../templatelist/templatelist.md)&gt; & | Specially formatted list of training input. Format should match what is passed to [project](#project-2).

* **output:** (void)


## void project(const [Template](../template/template.md) &src, [Template](../template/template.md) &dst) {: #project-1 }

This is a pure virtual function. It must be overloaded by all derived classes. Project a template through the transform, modifying its contents in some way and storing the modified data in **dst**. This function has a strict [Template](../template/template.md) in, [Template](../template/template.md) out model. For a multiple [Template](../template/template.md) in, multiple [Template](../template/template.md) out model see [project]{: #project-2 }.

* **function definition:**

        virtual void project(const Template &src, Template &dst) const = 0

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    src | const [Template](../template/template.md) & | Input template. It is immutable
    dst | [Template](../template/template.md) & | Output template. Should contain the modified data from the input template.

* **output:** (void)
* **example:**

        Template src("color_picture.jpg"), dst;

        Transform *color_converter = Transfrom::make("Read+Cvt(Gray)");
        color_converter->project(src, dst); // returns a grayscale image stored in dst


## void project(const [TemplateList](../templatelist/templatelist.md) &src, [TemplateList](../templatelist/templatelist.md) &dst) {: #project-2 }

This is a virtual function. Project multiple [Templates](../template/template.md) in and get multiple, modified, [Templates](../template/template.md) out. Especially useful in cases like detection where the requirement is image in, multiple objects out. The default implementation calls [project](#project-1) on each [Template](../template/template.md) in **src** and appends the results to **dst**.

* **function definition:**

        virtual void project(const TemplateList &src, TemplateList &dst) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    src | const [TemplateList](../templatelist/templatelist.md) & | Input templates. It is immutable
    dst | [TemplateList](../templatelist/templatelist.md) & | Output templates. Should contain the modified data from the input templates

* **output:** (void)
* **example:**

        TemplateList src(QList<Template>() << Template("image_with_faces.jpg")), dst;

        Transform *face_detector = Transform::make("FaceDetector");
        face_detector->project(src, dst); // dst will have one template for every face detected in src


## void projectUpdate(const [Template](../template/template.md) &src, [Template](../template/template.md) &dst) {: #projectupdate-1 }

This is a virtual function. Very similar to [project](#project-1) except this version is not **const** and can modify the internal state of the transform.

* **function definition:**

        virtual void projectUpdate(const Template &src, Template &dst)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    src | const [Template](../template/template.md) & | Input template. It is immutable
    dst | [Template](../template/template.md) & | Output template. Should contain the modified data from the input template.

* **output:** (void)
* **example:**

        class ExampleTransform : public Transform
        {
            Q_OBJECT

            int internalState;

            void projectUpdate(const Template &src, Template &dst)
            {
                dst = src;
                internalState++;
            }

            ...
        };

        BR_REGISTER(Transform, ExampleTransform)

        Template src("picture.jpg"), dst;

        Transform *example = Transform::make("Example");
        example->projectUpdate(src, dst); // dst is unchanged but Example's internalState has been incremented.


## void projectUpdate(const [TemplateList](../templatelist/templatelist.md) &src, [TemplateList](../templatelist/templatelist.md) &dst) {: #projectupdate-2 }

This is a virtual function. Very similar to [project](#project-2) except this version is not **const** and can modify the internal state of the transform.

* **function definition:**

        virtual void projectUpdate(const TemplateList &src, TemplateList &dst)

* **parameters:**

        Parameter | Type | Description
        --- | --- | ---
        src | const [TemplateList](../templatelist/templatelist.md) & | Input templates. It is immutable
        dst | [TemplateList](../templatelist/templatelist.md) & | Output templates. Should contain the modified data from the input templates

* **output:** (void)
* **example:**

        class ExampleTransform : public Transform
        {
            Q_OBJECT

            int internalState;

            void projectUpdate(const TemplateList &src, TemplateList &dst)
            {
                dst = src;
                internalState++;
            }

            ...
        };

        BR_REGISTER(Transform, ExampleTransform)

        TemplateList src(QList<Template>() << Template("picture.jpg")), dst;

        Transform *example = Transform::make("Example");
        example->projectUpdate(src, dst); // dst is unchanged but Example's internalState has been incremented.


## void projectUpdate([Template](../template/template.md) &srcdst) {: #projectupdate-3 }

In-place version of [projectUpdate](#projectupdate-1).

* **function definition:**

        void projectUpdate(Template &srcdst)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    srcdst | [Template](../template/template.md) & | Input and output template. It is overwritten with the output value after projecting

* **output:** (void)
* **example:**

        class ExampleTransform : public Transform
        {
            Q_OBJECT

            int internalState;

            void projectUpdate(const Template &src, Template &dst)
            {
                dst = src;
                internalState++;
            }

            ...
        };

        BR_REGISTER(Transform, ExampleTransform)

        Template src("picture.jpg");

        Transform *example = Transform::make("Example");
        example->projectUpdate(src); // src is modified in-place and Example's internalState has been incremented.


## void projectUpdate([TemplateList](../templatelist/templatelist.md) &srcdst) {: #projectupdate-4 }

In-place version of [projectUpdate](#projectupdate-2).

* **function definition:**

        void projectUpdate(TemplateList &srcdst)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    srcdst | [TemplateList](../templatelist/templatelist.md) & | Input and output templates. It is overwritten with the output value after projecting

* **output:** (void)
* **example:**

        class ExampleTransform : public Transform
        {
            Q_OBJECT

            int internalState;

            void projectUpdate(const TemplateList &src, TemplateList &dst)
            {
                dst = src;
                internalState++;
            }

            ...
        };

        BR_REGISTER(Transform, ExampleTransform)

        TemplateList src(QList<Template>() << Template("picture.jpg"));

        Transform *example = Transform::make("Example");
        example->projectUpdate(src); // src is modified in-place and Example's internalState has been incremented.


## bool timeVarying() {: #timevarying }

This is a virtual function. Check if the transform is time varying. Time varying means the internal state of the transform needs to be updated during projection. Time varying transforms should overload this function to return true and implement [projectUpdate](#projectupdate-1) instead of [project](#project-1).

* **function definition:**

        virtual bool timeVarying() const

* **parameters:** NONE
* **output:** (bool) Returns true if the transform is time varying and false otherwise


## [Template](../template/template.md) operator()(const [Template](../template/template.md) &src) {: #operator-pp-1 }

A convenience function to call [project](#project-1)

* **function definition:**

        inline Template operator()(const Template &src) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    src | const [Template](../template/template.md) & | Input template. It is immutable.

* **output**: ([Template](../template/template.md)) Returns the result of calling [project](#project-1)
* **example:**

        Template src("color_picture.jpg");

        Transform *color_converter = Transfrom::make("Read+Cvt(Gray)");
        Template dst = color_converter(src); // returns a grayscale image


## [TemplateList](../templatelist/templatelist.md) operator()(const [TemplateList](../templatelist/templatelist.md) &src) {: #operator-pp-2 }

A convenience function to call [project](#project-2)

* **function definition:**

        inline TemplateList operator()(const TemplateList &src) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    src | const [TemplateList](../templatelist/templatelist.md) & | Input templates. It is immutable.

* **output**: ([TemplateList](../templatelist/templatelist.md)) Returns the result of calling [project](#project-2)
* **example:**

        TemplateList src(QList<Template>() << Template("color_picture.jpg"));

        Transform *color_converter = Transfrom::make("Read+Cvt(Gray)");
        TemplateList dst = color_converter(src); // returns a list of grayscale images


## [Transform](transform.md) \*smartCopy(bool &newTransform) {: #smartcopy-1 }

This is a virtual function. Only [TimeVaryingTransforms](../timevaryingtransform/timevaryingtransform.md) need to overload this method. Similar to [clone](#clone), this function returns a deep copy of the transform. Unlike [clone](#clone), which copies everything, this function should copy the minimum amount required so that [projectUpdate](#projectupdate-1) can be called safely on the original instance and the copy returned by this function concurrently.

* **function definition:**

        virtual Transform *smartCopy(bool &newTransform)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    newTransform | bool & | True if the returned transform is newly allocated, false otherwise. This is used to handle deallocation. If newTransform is true, the caller of this function is responsible for deallocating it. If not, a [QSharedPointer][QSharedPointer] can wrap the output, and it will be deallocated elsewhere.

* **output:** ([Transform](transform.md) \*) Returns a pointer to a deep, smart, copy of the transform


## [Transform](transform.md) \*smartCopy() {: #smartcopy-2 }

Convenience function to call [smartCopy](#smartcopy-1) without arguments

* **function definition:**

        virtual Transform *smartCopy()

* **parameters:** NONE
* **output:** ([Transform](transform.md) \*) Returns a pointer, to a deep, smart, copy of the transform


## [Transform](transform.md) \*simplify(bool &newTransform) {: #simplify }

This is a virtual function. Get a simplified version of the transform for use at project time. The simplified version of the [Transform](transform.md) does not include any [Transforms](transform.md) that are only active at train time. It also removes any [LoadStore](../../../plugin_docs/core.md#loadstoretransform) transforms and keeps only their children. Transforms that only are active at train time (see [DownsampleTrainingTransform](../../../plugin_docs/core.md#downsampletrainingtransform) as an example) should overload this function and return their children if they have any or NULL if they do not.

* **function definition:**

        virtual Transform * simplify(bool &newTransform)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    newTransform | bool & | True if the simplified transform is newly allocated, false otherwise. If true, the caller of this function is responsible for deallocating the transform.

* **output:** ([Transform](transform.md) \*) Returns a pointer to the simplified version of the transform
* **example:**

        Transform *transform = Transform::make("DataProcessing+DownsampleTraining(Example)+<ModelFile>");
        transfrom->description(); // returns "DataProcessing+DownsampleTraining(Example)+LoadStore(transformString=TransformFromModelFile)"

        bool newTransform;
        transform->simplify(newTransform)->description(); // returns "DataProcessing+Example+TransformFromModelFile"


## [Transform](transform.md) \*make(const [QString][QString] &description) {: #make }

This is a protected function. Makes a child transform from a provided description by calling [make](statics.md#make) with parent = <tt>this</tt>.

* **function definition:**

        inline Transform *make(const QString &description)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    description | const [QString][QString] & | Description of the child transform

* **output:** ([Transform](transform.md) \*) Returns a pointer to the created child transform

<!-- Links -->
[Mat]: http://docs.opencv.org/modules/core/doc/basic_structures.html#mat "Mat"
[QList]: http://doc.qt.io/qt-5/QList.html "QList"
[QString]: http://doc.qt.io/qt-5/QString.html "QString"
[QSharedPointer]: http://doc.qt.io/qt-5/qsharedpointer.html "QSharedPointer"
