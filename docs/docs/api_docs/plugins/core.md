# CacheTransform

Caches br::Transform::project() results.

* **file:** core/cache.cpp
* **inherits:** [MetaTransform](../cpp_api/metatransform/metatransform.md)
* **author:** Josh Klontz
* **properties:** None


---

# CollectOutputTransform

DOCUMENT ME CHARLES

* **file:** core/stream.cpp
* **inherits:** [TimeVaryingTransform](../cpp_api/timevaryingtransform/timevaryingtransform.md)
* **author:** Charles Otto
* **properties:** None


---

# ContractTransform

It's like the opposite of ExpandTransform, but not really

* **file:** core/contract.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author:** Charles Otto
* **properties:** None


---

# CrossValidateTransform

Cross validate a trainable transform.

* **file:** core/crossvalidate.cpp
* **inherits:** [MetaTransform](../cpp_api/metatransform/metatransform.md)
* **authors:** Josh Klontz, Scott Klum
* **properties:** None


---

# DirectStreamTransform

DOCUMENT ME CHARLES

* **file:** core/stream.cpp
* **inherits:** [CompositeTransform](#compositetransform)
* **author:** Charles Otto
* **properties:** None


---

# DiscardTransform

Removes all template's matrices.

* **file:** core/discard.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **see:** [IdentityTransform FirstTransform RestTransform RemoveTransform](#identitytransform firsttransform resttransform removetransform)
* **author:** Josh Klontz
* **properties:** None


---

# DownsampleTrainingTransform

DOCUMENT ME JOSH

* **file:** core/downsampletraining.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author:** Josh Klontz
* **properties:** None


---

# ExpandTransform

Performs an expansion step on input templatelists

* **file:** core/expand.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **see:** [PipeTransform](#pipetransform)
* **author:** Josh Klontz
* **properties:** None


---

# FTETransform

Flags images that failed to enroll based on the specified transform.

* **file:** core/fte.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author:** Josh Klontz
* **properties:** None


---

# FirstTransform

Removes all but the first matrix from the template.

* **file:** core/first.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **see:** [IdentityTransform DiscardTransform RestTransform RemoveTransform](#identitytransform discardtransform resttransform removetransform)
* **author:** Josh Klontz
* **properties:** None


---

# ForkTransform

Transforms in parallel.

* **file:** core/fork.cpp
* **inherits:** [CompositeTransform](#compositetransform)
* **see:** [PipeTransform](#pipetransform)
* **author:** Josh Klontz
* **properties:** None


---

# GalleryCompareTransform

Compare each template to a fixed gallery (with name = galleryName), using the specified distance.

* **file:** core/gallerycompare.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author:** Charles Otto
* **properties:** None


---

# IdentityTransform

A no-op transform.

* **file:** core/identity.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **see:** [DiscardTransform FirstTransform RestTransform RemoveTransform](#discardtransform firsttransform resttransform removetransform)
* **author:** Josh Klontz
* **properties:** None


---

# IndependentTransform

Clones the transform so that it can be applied independently.

* **file:** core/independent.cpp
* **inherits:** [MetaTransform](../cpp_api/metatransform/metatransform.md)
* **author:** Josh Klontz
* **properties:** None


---

# LikelyTransform

Generic interface to Likely JIT compiler

* **file:** core/likely.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author:** Josh Klontz
* **properties:** None


---

# LoadStoreTransform

Caches transform training.

* **file:** core/loadstore.cpp
* **inherits:** [MetaTransform](../cpp_api/metatransform/metatransform.md)
* **author:** Josh Klontz
* **properties:** None


---

# PipeTransform

Transforms in series.

* **file:** core/pipe.cpp
* **inherits:** [CompositeTransform](#compositetransform)
* **see:**

	* [ExpandTransform](ExpandTransform)
	* [ForkTransform](ForkTransform)

* **author:** Josh Klontz
* **properties:** None


---

# ProcessWrapperTransform

Interface to a separate process

* **file:** core/processwrapper.cpp
* **inherits:** [WrapperTransform](#wrappertransform)
* **author:** Charles Otto
* **properties:** None


---

# ProcrustesAlignTransform

Improved procrustes alignment of points, to include a post processing scaling of points

* **file:** core/align.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author:** Brendan Klare
* **properties:** None


---

# RestTransform

Removes the first matrix from the template.

* **file:** core/rest.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **see:** [IdentityTransform DiscardTransform FirstTransform RemoveTransform](#identitytransform discardtransform firsttransform removetransform)
* **author:** Josh Klontz
* **properties:** None


---

# SchrodingerTransform

Generates two templates, one of which is passed through a transform and the other

* **file:** core/schrodinger.cpp
* **inherits:** [MetaTransform](../cpp_api/metatransform/metatransform.md)
* **author:** Scott Klum
* **properties:** None


---

# SingletonTransform

A globally shared transform.

* **file:** core/singleton.cpp
* **inherits:** [MetaTransform](../cpp_api/metatransform/metatransform.md)
* **author:** Josh Klontz
* **properties:** None


---

# StreamTransform

DOCUMENT ME CHARLES

* **file:** core/stream.cpp
* **inherits:** [WrapperTransform](#wrappertransform)
* **author:** Charles Otto
* **properties:** None


---

# SynthesizePointsTransform

Synthesize additional points via triangulation.

* **file:** core/align.cpp
* **inherits:** [MetadataTransform](../cpp_api/metadatatransform/metadatatransform.md)
* **author:** Josh Klontz
* **properties:** None


---

# TextureMapTransform

Maps texture from one set of points to another. Assumes that points are rigidly transformed

* **file:** core/align.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **authors:** Brendan Klare, Scott Klum
* **properties:** None


---

