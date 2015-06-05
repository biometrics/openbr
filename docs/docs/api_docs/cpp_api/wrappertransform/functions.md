## bool timeVarying() {: #timevarying }

Check whether the transform is timeVarying.

* **function definition:**

        bool timeVarying() const

* **parameters:** NONE
* **output:** (bool) Returns true if the [child transform](properties.md) is time varying, false otherwise

## void train(const [QList][QList]&lt;[TemplateList][../templatelist/templatelist.md)&gt; &data) {: #train }

Call train on the child transform

* **function defintion:**

        void train(const QList<TemplateList> &data)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    data | const [QList][QList]&lt;[TemplateList](../templatelist/templatelist.md)&gt; & | The training data

* **output:** (void)

## void project(const [Template](../template/template.md) &src, [Template](../template/template.md) &dst) {: #project-1 }

Call project on the child transform

* **function definition:**

        void project(const Template &src, Template &dst) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    src | const [Template](../template/template.md) & | The input template
    dst | [Template](../template/template.md) & | The output template

* **output:** (void)

## void project(const [TemplateList](../templatelist/templatelist.md) &src, [TemplateList](../templatelist/templatelist.md) &dst) {: #project-2 }

Call project on the child transform

* **function definition:**

        void project(const TemplateList &src, TemplateList &dst) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    src | const [TemplateList](../templatelist/templatelist.md) & | The input template list
    dst | [TemplateList](../templatelist/templatelist.md) & | The output template list

* **output:** (void)

## void projectUpdate(const [Template](../template/template.md) &src, [Template](../template/template.md) &dst) {: #projectupdate-1 }

Call projectUpdate on the child transform

* **function definition:**

        void projectUpdate(const Template &src, Template &dst)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    src | const [Template](../template/template.md) & | The input template
    dst | [Template](../template/template.md) & | The output template

* **output:** (void)

## void projectUpdate(const [TemplateList](../templatelist/templatelist.md) &src, [TemplateList](../templatelist/templatelist.md) &dst) {: #projectupdate-2 }

Call projectUpdate on the child transform

* **function definition:**

        void projectUpdate(const TemplateList &src, TemplateList &dst)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    src | const [TemplateList](../templatelist/templatelist.md) & | The input template list
    dst | [TemplateList](../templatelist/templatelist.md) & | The output template list

* **output:** (void)

## void init() {: #init }

Initialize the transform. Sets [trainable](../transform/members.md#trainable) to match the child transform (if the child is trainable so is the wrapper)

* **function definition:**

        void init()

* **parameters:** NONE
* **output:** (void)

## void finalize([TemplateList](../templatelist/templatelist.md) &output) {: #finalize }

This is a virtual function. Call finalize on the child transform

* **function definition:**

        virtual void finalize(TemplateList &output)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    output | const [TemplateList](../templatelist/templatelist.md) & | The output to finalize

* **output:** (void)

## [Transform](../transform/transform.md) \*simplify(bool &newTransform) {: #simplify }

This is a virtual function. Calls simplify on the child transform.

* **function definition:**

        virtual Transform *simplify(bool &newTransform)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    newTransform | bool & | True if a new, simplified, transform was allocated inside this call, false otherwise

* **output:** ([Transform](../transform/transform.md) \*) Returns itself if the child transform cannot be simplified. newTransform is set to false in this case. If the child can be simplified, a new WrapperTransform is allocated with the child transform set as the simplified version of the old child transform. newTransform is set to true in this case

## [Transform](../transform/transform.md) \*smartCopy(bool &newTransform) {: #smartcopy }

Get a smart copy, meaning a copy only if one is required, of this transform

* **function definition:**

        Transform *smartCopy(bool &newTransform)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    newTransform | bool & | True if a new, simplified, transform was allocated inside this call, false otherwise

* **output:** ([Transform](../transform/transform.md) \*) Returns itself if the child transform is not time varying (no copy needed). newTransform is set to false in this case. If the child is time varying make a copy by calling [smartCopy](../timevaryingtransform/functions.md#smartcopy) on the child. newTransform is set to true in this case.

<!-- Links -->
[QList]: http://doc.qt.io/qt-5/QList.html "QList"
