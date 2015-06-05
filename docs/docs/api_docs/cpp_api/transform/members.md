Member | Type | Description
--- | --- | ---
<a class="table-anchor" id=independent></a>independent | bool | True if the transform is independent, false otherwise. Independent transforms process each [Mat][Mat] in a [Template](../template/template.md) independently. This means that a new instance of the transform is created for each [Mat][Mat]. If the transform is [trainable](#trainable) and the training data has more then one [Mat][Mat] per template, each created instance of the transform is trained separately. Please see [Training Algorithms](../../../tutorials.md#training-algorithms) for more details.
<a class="table-anchor" id=trainable></a>trainable | bool | True if the transform is trainable, false otherwise. Trainable transforms need to overload the [train](functions.md#train-1) function.

<!-- Links -->
[Mat]: http://docs.opencv.org/modules/core/doc/basic_structures.html#mat "Mat"
