## bool timeVarying() {: #timevarying }

Check if the transform is time varying. The transform is time varying if any of its children are time varying.

* **function definition:**

		bool timeVarying() const

* **parameters:** NONE
* **output:** (bool) Returns [isTimeVarying](members.md#istimevarying)

## void init() {: #init }

Initialize the transform. If any of the child transform are time varying, [isTimeVarying](members.md#istimevarying) is set to true. Similarly if any of the child transforms are trainable, [trainable](../transform/members.md#trainable) is set to true.

* **function definition:**

		void init()

* **parameters:** NONE
* **output:** (void)

## void project(const [Template](../template/template.md) &src, [Template](../template/template.md) &dst) {: #project-1 }

If the transform is time varying call [timeInvariantAlias](../timevaryingtransform/members.md#timeinvariantalias) project, which ensures thread safety. If the transform is not time varying call [_project](#_project-1).

* **function definition:**

		virtual void project(const Template &src, Template &dst) const

* **parameters:**

	Parameter | Type | Description
	--- | --- | ---
	src | const [Template](../template/template.md) & | The input template
	dst | [Template](../template/template.md) & | The output template

* **output:** (void)

## void project(const [TemplateList](../templatelist/templatelist.md) &src, [TemplateList](../templatelist/templatelist.md) &dst) {: #project-2 }

If the transform is time varying call [timeInvariantAlias](../timevaryingtransform/members.md#timeinvariantalias) project, which ensures thread safety. If the transform is not time varying call [_project](#_project-2).

* **function definition:**

		virtual void project(const TemplateList &src, TemplateList &dst) const

* **parameters:**

	Parameter | Type | Description
	--- | --- | ---
	src | const [TemplateList](../templatelist/templatelist.md) & | The input template list
	dst | [TemplateList ](../templatelist/templatelist.md) & | The output template list

* **output:** (void)

## void \_project(const [Template](../template/template.md) &src, [Template](../template/template.md) &dst) {: #\_project-1 }

This is a pure virtual function. It should handle standard projection through the child transforms

* **function definition:**

		virtual void _project(const Template &src, Template &dst) const = 0

* **parameters:**

	Parameter | Type | Description
	--- | --- | ---
	src | const [Template](../template/template.md) & | The input template
	dst | [Template](../template/template.md) & | The output template

* **output:** (void)

## void \_project(const [TemplateList](../templatelist/templatelist.md) &src, [TemplateList](../templatelist/templatelist.md) &dst) {: #\_project-2 }

This is a pure virtual function. It should handle standard projection through the child transforms

* **function definition:**

		virtual void _project(const TemplateList &src, TemplateList &dst) const = 0

* **parameters:**

	Parameter | Type | Description
	--- | --- | ---
	src | const [TemplateList](../templatelist/templatelist.md) & | The input template list
	dst | [TemplateList](../templatelist/templatelist.md) & | The output template list

* **output:** (void)

## [Transform](../transform/transform.md) \*simplify(bool &newTransform) {: #simplify }

This is a virtual function. Calls [simplify](../transform/functions.md#simplify) on each child transform.

* **function definition:**

		virtual Transform *simplify(bool &newTransform)

* **parameters:**

	Parameter | Type | Description
	--- | --- | ---
	newTransform | bool & | True if a new, simplified, transform was allocated inside this call, false otherwise

* **output:** ([Transform](../transform/transform.md) \*) Returns itself if none of the children can be simplified. newTransform is false in this case. If any child can be simplified a new [CompositeTransform](compositetransform.md) is allocated and its children are set as the result of calling simplify on each of the old children. newTransform is true in this case

## [Transform](../transform/transform.md) \*smartCopy(bool &newTransform) {: #smartcopy }

Get a smart copy, meaning a copy only if one is required, of this transform

* **function definition:**

		Transform *smartCopy(bool &newTransform)

* **parameters:**

	Parameter | Type | Description
	--- | --- | ---
	newTransform | bool & | True if a new, simplified, transform was allocated inside this call, false otherwise

* **output:** ([Transform](../transform/transform.md) \*) Returns itself if [isTimeVarying](members.md#istimevarying) is false (no copy needed). newTransform is set to false in this case. If [isTimeVarying](members.md#istimevarying) is true, a new [CompositeTransform](compositetransform.md) is allocated and its children are set to the result of calling smartCopy on each of the old children. newTransform is true in this case.
