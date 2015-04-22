<!-- META TRANSFORM -->

Inherits [Transform](../transform/transform.md)

A [Transform](../transform/transform.md) that expect multiple matrices per [Template](../template/template.md)

See:

* [Members](members.md)
* [Constructors](constructors.md)
* [Static Functions](statics.md)
* [Functions](functions.md)

[MetaTransforms](metatransform.md) are [Transforms](../transform/transform.md) that are not [independent](../transform/members.md#independent). This means they expect more then one matrix in each input [Template](../template/template.md) and a new instance of the transform should not be created for each matrix. 
