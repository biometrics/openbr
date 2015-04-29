<!-- UNTRAINABLE METADATA TRANSFORM -->

Inherits [MetadataTransform](../metatransform/metatransform.md)

A [Transform](../transform/transform.md) that requires only [Template](../template/template.md) [metadata](../template/members.md#file) and does not require training.

See:

* [Constructors](constructors.md)

[UntrainableMetadataTransforms](untrainablemetadatatransform.md) are [Transforms](../transform/transform.md) that operate soley on textual [metadata](../template/members.md#file). They are *not* [independent](../transform/members.md#independent), because [Templates](../template/template.md) can only every have one collection of [metadata](../template/members.md#file), and they are *not* [trainable](../transform/members.md#trainable).
