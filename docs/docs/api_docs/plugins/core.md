# CacheTransform

Caches [Transform](../cpp_api/transform/transform.md)::project() results.
 

* **file:** core/cache.cpp
* **inherits:** [MetaTransform](../cpp_api/metatransform/metatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# CollectOutputTransform

DOCUMENT ME CHARLES
 

* **file:** core/stream.cpp
* **inherits:** [TimeVaryingTransform](../cpp_api/timevaryingtransform/timevaryingtransform.md)
* **author(s):** [Charles Otto][caotto]
* **properties:** None


---

# ContractTransform

It's like the opposite of ExpandTransform, but not really

Given a [TemplateList](../cpp_api/templatelist/templatelist.md) as input, concatenate them into a single [Template](../cpp_api/template/template.md)
 

* **file:** core/contract.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Charles Otto][caotto]
* **properties:** None


---

# CrossValidateTransform

Cross validate a trainable [Transform](../cpp_api/transform/transform.md).

To use an extended [Gallery](../cpp_api/gallery/gallery.md), add an allPartitions="true" flag to the gallery sigset for those images that should be compared
against for all testing partitions.
 

* **file:** core/crossvalidate.cpp
* **inherits:** [MetaTransform](../cpp_api/metatransform/metatransform.md)
* **author(s):** [Josh Klontz][jklontz], [Scott Klum][sklum]
* **properties:** None


---

# DirectStreamTransform

DOCUMENT ME CHARLES
 

* **file:** core/stream.cpp
* **inherits:** [CompositeTransform](../cpp_api/compositetransform/compositetransform.md)
* **author(s):** [Charles Otto][caotto]
* **properties:** None


---

# DiscardTemplatesTransform

DOCUMENT ME
 

* **file:** core/discardtemplates.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Unknown][unknown]
* **properties:** None


---

# DiscardTransform

Removes all matrices from a [Template](../cpp_api/template/template.md).
 

* **file:** core/discard.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# DistributeTemplateTransform

DOCUMENT ME
 

* **file:** core/distributetemplate.cpp
* **inherits:** [MetaTransform](../cpp_api/metatransform/metatransform.md)
* **author(s):** [Unknown][unknown]
* **properties:** None


---

# DownsampleTrainingTransform

DOCUMENT ME JOSH
 

* **file:** core/downsampletraining.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# EventTransform

DOCUMENT ME
 

* **file:** core/event.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Unknown][unknown]
* **properties:** None


---

# ExpandTransform

Performs an expansion step on an input [TemplateList](../cpp_api/templatelist/templatelist.md). Each matrix in each input [Template](../cpp_api/template/template.md) is expanded into its own [Template](../cpp_api/template/template.md).
 

* **file:** core/expand.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# FTETransform

Flags images that failed to enroll based on the specified [Transform](../cpp_api/transform/transform.md).
 

* **file:** core/fte.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# FirstTransform

Removes all but the first matrix from the [Template](../cpp_api/template/template.md).
 

* **file:** core/first.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# ForkTransform

Transforms in parallel.

The source [Template](../cpp_api/template/template.md) is seperately given to each transform and the results are appended together.
 

* **file:** core/fork.cpp
* **inherits:** [CompositeTransform](../cpp_api/compositetransform/compositetransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# GalleryCompareTransform

Compare each [Template](../cpp_api/template/template.md) to a fixed [Gallery](../cpp_api/gallery/gallery.md) (with name = galleryName), using the specified distance.
dst will contain a 1 by n vector of scores.
 

* **file:** core/gallerycompare.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author(s):** [Charles Otto][caotto]
* **properties:** None


---

# IdentityTransform

A no-op [Transform](../cpp_api/transform/transform.md).
 

* **file:** core/identity.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# IndependentTransform

Clones the [Transform](../cpp_api/transform/transform.md) so that it can be applied independently.

Independent [Transform](../cpp_api/transform/transform.md)s expect single-matrix [Template](../cpp_api/template/template.md).
 

* **file:** core/independent.cpp
* **inherits:** [MetaTransform](../cpp_api/metatransform/metatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# JNITransform

Execute Java code from OpenBR using the JNI
 

* **file:** core/jni.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Jordan Cheney][jcheney]
* **properties:** None


---

# LikelyTransform

Generic interface to Likely JIT compiler
 

* **file:** core/likely.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **see:** [www.liblikely.org](www.liblikely.org)
* **properties:** None


---

# LoadStoreTransform

Caches [Transform](../cpp_api/transform/transform.md) training.
 

* **file:** core/loadstore.cpp
* **inherits:** [MetaTransform](../cpp_api/metatransform/metatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# PipeTransform

Transforms in series.

The source [Template](../cpp_api/template/template.md) is given to the first transform and the resulting [Template](../cpp_api/template/template.md) is passed to the next transform, etc.
 

* **file:** core/pipe.cpp
* **inherits:** [CompositeTransform](../cpp_api/compositetransform/compositetransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# ProcessWrapperTransform

Interface to a separate process
 

* **file:** core/processwrapper.cpp
* **inherits:** [WrapperTransform](../cpp_api/wrappertransform/wrappertransform.md)
* **author(s):** [Charles Otto][caotto]
* **properties:** None


---

# ProcrustesAlignTransform

Improved procrustes alignment of points, to include a post processing scaling of points
to faciliate subsequent texture mapping.
 

* **file:** core/align.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author(s):** [Brendan Klare][bklare]
* **properties:**

	Property | Type | Description
	--- | --- | ---
	width | float | Width of output coordinate space (before padding)
	padding | float | Amount of padding around the coordinate space
	useFirst | bool | Whether or not to use the first instance as the reference object

---

# ProgressCounterTransform

DOCUMENT ME
 

* **file:** core/progresscounter.cpp
* **inherits:** [TimeVaryingTransform](../cpp_api/timevaryingtransform/timevaryingtransform.md)
* **author(s):** [Unknown][unknown]
* **properties:** None


---

# RemoveTransform

Removes the matrix from the [Template](../cpp_api/template/template.md) at the specified index.
 

* **file:** core/remove.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# RestTransform

Removes the first matrix from the [Template](../cpp_api/template/template.md).
 

* **file:** core/rest.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# SchrodingerTransform

Generates two [Template](../cpp_api/template/template.md), one of which is passed through a [Transform](../cpp_api/transform/transform.md) and the other
is not. No cats were harmed in the making of this [Transform](../cpp_api/transform/transform.md).
 

* **file:** core/schrodinger.cpp
* **inherits:** [MetaTransform](../cpp_api/metatransform/metatransform.md)
* **author(s):** [Scott Klum][sklum]
* **properties:** None


---

# SingletonTransform

A globally shared [Transform](../cpp_api/transform/transform.md).
 

* **file:** core/singleton.cpp
* **inherits:** [MetaTransform](../cpp_api/metatransform/metatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# StreamTransform

DOCUMENT ME CHARLES
 

* **file:** core/stream.cpp
* **inherits:** [WrapperTransform](../cpp_api/wrappertransform/wrappertransform.md)
* **author(s):** [Charles Otto][caotto]
* **properties:** None


---

# SynthesizePointsTransform

Synthesize additional points via triangulation.
 

* **file:** core/align.cpp
* **inherits:** [MetadataTransform](../cpp_api/metadatatransform/metadatatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# TextureMapTransform

Maps texture from one set of points to another. Assumes that points are rigidly transformed
 

* **file:** core/align.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Brendan Klare][bklare], [Scott Klum][sklum]
* **properties:** None


---

