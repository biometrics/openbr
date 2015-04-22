<!-- UNTRAINABLE METATRANSFORM -->

Inherits [UntrainableTransform](../untrainabletransform/untrainabletransform.md)

A [Transform](../transform/transform.md) that expect multiple matrices per [Template](../template/template.md) and is not [trainable](../transform/members.md#trainable)

See:

* [Members](members.md)
* [Constructors](constructors.md)
* [Static Functions](statics.md)
* [Functions](functions.md)

[UntrainableMetaTransforms](untrainablemetatransform.md) are [UntrainableTransforms](../untrainabletransform/untrainabletransform.md) that are not [independent](../transform/members.md#independent). This means they expect more then one matrix in each input [Template](../template/template.md) and a new instance of the transform should not be created for each matrix. 
