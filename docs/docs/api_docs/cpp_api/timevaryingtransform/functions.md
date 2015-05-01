## bool timeVarying() {: #timevarying }

This is a virtual function. Check if the transform is time varying. This always evaluates to true.

* **function definition:**

        virtual bool timeVarying() const

* **parameters:** NONE
* **output:** (bool) Returns true (the transform is always time varying)

## void project(const [Template](../template/template.md) &src, [Template](../template/template.md) &dst) {: #project-1 }

This is a virtual function. For [TimeVaryingTransforms](timevaryingtransform.md) normal enrollment calls [projectUpdate](#projectupdate-2). It is still possible to call this version of project instead but it must be done explicitly and is **strongly** discouraged. If this function is called [timeInvariantAlias](members.md#timeinvariantalias) is used to call [projectUpdate](#projectupdate-2) in a thread safe way.

* **function definition:**

        virtual void project(const Template &src, Template &dst) const

 * **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    src | const [Template](../template/template.md) & | The input template
    dst | [Template](../template/template.md) & | The output template

* **output:** (void)

## void project(const [TemplateList](../templatelist/templatelist.md) &src, [TemplateList](../templatelist/templatelist.md) &dst) {: #project-2 }

This is a virtual function. For [TimeVaryingTransforms](timevaryingtransform.md) normal enrollment calls [projectUpdate](#projectupdate-2). It is still possible to call this version of project instead but it must be done explicitly and is **strongly** discouraged. If this function is called [timeInvariantAlias](members.md#timeinvariantalias) is used to call [projectUpdate](#projectupdate-2) in a thread safe way.

* **function definition:**

        virtual void project(const TemplateList &src, TemplateList &dst) const

 * **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    src | const [TemplateList](../templatelist/templatelist.md) & | The input template list
    dst | [TemplateList](../templatelist/templatelist.md) & | The output template list

* **output:** (void)

## void projectUpdate(const [Template](../template/template.md) &src, [Template](../template/template.md) &dst) {: #projectupdate-1 }

This is a virtual function. This function should never be called because it is useless to implement a Template -> Template call using project update. An error is thrown if it is called.

* **function definition:**

        virtual void projectUpdate(const Template &src, Template &dst)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    src | const [Template](../template/template.md) & | The input template
    dst | [Template](../template/template.md) & | The output template

* **output:** (void)

## void projectUpdate(const [Template](../template/template.md) &src, [Template](../template/template.md) &dst) {: #projectupdate-1 }

This is a virtual function. This function should never be called because it is useless to implement a Template -> Template call using project update. An error is thrown if it is called.

* **function definition:**

        virtual void projectUpdate(const Template &src, Template &dst)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    src | const [Template](../template/template.md) & | The input template
    dst | [Template](../template/template.md) & | The output template

* **output:** (void)

## void projectUpdate(const [Template](../template/template.md) &src, [Template](../template/template.md) &dst) {: #projectupdate-1 }

This is a virtual function. This function should never be called because it is useless to implement a Template -> Template call using project update. An error is thrown if it is called.

* **function definition:**

        virtual void projectUpdate(const Template &src, Template &dst)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    src | const [Template](../template/template.md) & | The input template
    dst | [Template](../template/template.md) & | The output template

* **output:** (void)

## void projectUpdate(const [TemplateList](../templatelist/templatelist.md) &src, [TemplateList](../templatelist/templatelist.md) &dst) {: #projectupdate-2 }

This is a virtual function. This is the non-const alternative to [project](../transform/functions.md#project-1). It allows the internal state of the transform to be update at project time.

* **function definition:**

        virtual void projectUpdate(const TemplateList &src, TemplateList &dst)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    src | const [TemplateList](../templatelist/templatelist.md) & | The input template list
    dst | [TemplateList](../templatelist/templatelist.md) & | The output template list

* **output:** (void)

## [Transform](../transform/transform.md) \*smartCopy(bool &newTransform) {: #smartcopy }

This is a virtual function. Make a smart copy of the transform.

* **function definition:**

        virtual Transform *smartCopy(bool &newTransform)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    newTransform | bool & | True if a new, simplified, transform was allocated inside this call, false otherwise

* **output:** ([Transform](../transform/transform.md) \*) Returns a copy of itself by default. newTransform is set to true in this case.
