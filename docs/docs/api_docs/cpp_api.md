# C++ Plugin API

The C++ Plugin API is a pluggable API designed to allow the succinct expression of biometrics algorithms as a series of independent plugins. The API exposes a number of data structures and plugin abstractions to use as simple building blocks for complex algorithms.

## Data Structures

The API defines two data structures:

* [File](cpp_api/file/file.md)s are typically used to store the path to a file on disk with associated metadata (in the form of key-value pairs). In the example above, we store the rectangles detected by [Cascade](../plugin_docs/metadata.md#cascadetransform) as metadata which are then used by [Draw](../plugin_docs/gui.md#drawtransform) for visualization.
* [Template](cpp_api/template/template.md)s are containers for images and [File](cpp_api/file/file.md)s. Images in OpenBR are OpenCV Mats and are member variables of Templates. Templates can contain one or more images.

## Plugin Abstractions

A plugin in OpenBR is defined as *classes which modify, interpret, or assist with the modification of OpenBR compatible images and/or metadata*. All OpenBR plugins have a common ancestor, [Object](cpp_api/object/object.md), and they all are constructed from string descriptions given by [Files](cpp_api/file/file.md). There are eight base abstractions that derive from [Object](cpp_api/object/object.md), one of which should be the parent for all production plugins. They are:

Plugin | Function
--- | ---
[Initializer](cpp_api/initializer/initializer.md) | Initializes shared contexts and variables at the launch of OpenBR. Typically used with third-party plugins.
[Transform](cpp_api/transform/transform.md) | The most common plugin type in OpenBR. Provides the basis for algorithms by transforming images or metadata.
[Distance](cpp_api/distance/distance.md) | Used to compute a distance metric between [Template](cpp_api/template/template.md)s.
[Format](cpp_api/format/format.md) | Used for I/O. Formats handle file types that correspond to single objects (e.g. .jpg, .png, etc.).
[Gallery](cpp_api/gallery/gallery.md) | Used for I/O. Galleries handle file types that correspond to many objects (e.g. .csv., .xml, etc.).
[Output](cpp_api/output/output.md) | Used for I/O. Outputs handle the results of [Distance](cpp_api/distance/distance.md) comparisons.
[Representation](cpp_api/representation/representation.md) | Converts images into feature vectors. Slightly different then [Transform](cpp_api/transform/transform.md)s in implementation and available API calls.
[Classifier](cpp_api/classifier/classifier.md) | Classifies feature vectors as members of specific classes or performs regression on them.

 Additionally, there are several child-abstractions for specific use cases. A few examples are:

 Plugin | Parent | Function
 --- | --- | ---
 [UntrainableTransform](cpp_api/untrainabletransform/untrainabletransform.md) | [Transform](cpp_api/transform/transform.md) | A [Transform](cpp_api/transform/transform.md) that cannot be trained.
 [MetaTransform](cpp_api/metatransform/metatransform.md) | [Transform](cpp_api/transform/transform.md)  | A [Transform](cpp_api/transform/transform.md) that is *not* [independent](cpp_api/transform/members.md#independent).
 [UntrainableMetaTransform](cpp_api/untrainablemetatransform/untrainablemetatransform.md) | [UntrainableTransform](cpp_api/untrainabletransform/untrainabletransform.md) | A [Transform](cpp_api/transform/transform.md) that is *not* [independent](cpp_api/transform/members.md#independent) and cannot be trained.
 [MetadataTransform](cpp_api/metadatatransform/metadatatransform.md) | [Transform](cpp_api/transform/transform.md) | A [Transform](cpp_api/transform/transform.md) that operates only on [Template](cpp_api/template/template.md) [metadata](cpp_api/template/members.md#file).
 [TimeVaryingTransform](cpp_api/timevaryingtransform/timevaryingtransform.md) | [Transform](cpp_api/transform/transform.md) | A [Transform](cpp_api/transform/transform.md) that changes at runtime as a result of the input.
 [UntrainableDistance](cpp_api/untrainabledistance/untrainabledistance.md) | [Distance](cpp_api/distance/distance.md) | A [Distance](cpp_api/distance/distance.md) that cannot be trained.
 [ListDistance](cpp_api/listdistance/listdistance.md) | [Distance](cpp_api/distance/distance.md) | A [Distance](cpp_api/distance/distance.md) with a list of child distances. The ListDistance is trainable is any of its children are trainable.
 [MatrixOutput](cpp_api/matrixoutput/matrixoutput.md) | [Output](cpp_api/output/output.md) | A [Output](cpp_api/output/output.md) that outputs data as a matrix.

 As previously mentioned, all plugins in OpenBR are constructed using strings stored as [Files](cpp_api/file/file.md). The construction is done at runtime by a [Factory](cpp_api/factory/factory.md) class[^1]. The [Factory](cpp_api/factory/factory.md) expects strings of the form `PluginName(property1=value1,property2=value2...propertyN=valueN)` (value lists are also permitted). It looks up the `PluginName` in a static registry and builds the plugin if found. The registry is populated using a special macro [BR_REGISTER](cpp_api/factory/macros.md#br_register); each plugin needs to register itself and its base abstraction in the factory to enable construction. The purpose of this is to allow algorithms to be described completely by strings. For more information on algorithms in OpenBR please see the [tutorial](../tutorials.md#algorithms-in-openbr)

[^1]: [Industrial Strength Pluggable Factories](https://adtmag.com/articles/2000/09/25/industrial-strength-pluggable-factories.aspx)
