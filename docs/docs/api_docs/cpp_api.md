# C++ Plugin API

The C++ Plugin API is a pluggable API designed to allow the succinct expression of biometrics algorithms as a series of independent plugins. The API exposes a number of data structures and plugin abstractions to use as simple building blocks for complex algorithms. 

## Data Structures

The API defines two data structures: 

* [Files](cpp_api/file/file.md)
* [Templates](cpp_api/template/template.md)

[Files](cpp_api/file/file.md) are used for storing textual data and [Templates](cpp_api/template/template.md) act as containers that hold images, stored as OpenCV [Mats][Mat], and [Files](cpp_api/file/file.md). 

## Plugin Abstractions

A plugin in OpenBR is defined as *classes which modify, interpret, or assist with the modification of OpenBR compatible images and/or metadata*. All OpenBR plugins have a common ancestor, [Object](cpp_api/object/object.md), and they all are constructed from string descriptions given by [Files](cpp_api/file/file.md). There are 8 base abstractions that derive from [Object](cpp_api/object/object.md) and should be the parent for all production plugins. They are:

Plugin | Function
--- | ---
[Initializer](cpp_api/initializer/initializer.md) | Initializes shared contexts and variables at the launch of OpenBR. Mostly useful for 3rdparty plugin additions.
[Transform](cpp_api/transform/transform.md) | The most common plugin type in OpenBR. Transforms images or metadata.
[Distance](cpp_api/distance/distance.md) | Finds the distance between templates.
[Format](cpp_api/format/format.md) | Used for I/O. Formats handle output types that correspond to single objects, for example .jpg, .png etc.
[Gallery](cpp_api/gallery/gallery.md) | Used for I/O. Galleries handle output types that correspond to many objects, for example .csv. .xml etc.
[Output](cpp_api/output/output.md) | Used for I/O. Outputs handle the results of Distance comparisons.
[Representation](cpp_api/representation/representation.md) | Converts images into feature vectors. Slightly different then Transforms in implementation and available API calls.
[Classifier](cpp_api/classifier/classifier.md) | Classifies feature vectors as members of specific classes
 
 Additionally, there are several child-abstractions for specific use cases. They are:
 
 Plugin | Parent | Function
 --- | --- | ---
 [UntrainableTransform](cpp_api/untrainabletransform/untrainabletransform.md) | [Transform](cpp_api/transform/transform.md) | A [Transform](cpp_api/transform/transform.md) that cannot be trained
 [MetaTransform](cpp_api/metatransform/metatransform.md) | [Transform](cpp_api/transform/transform.md)  | A [Transform](cpp_api/transform/transform.md) that is *not* [independent](cpp_api/transform/members.md#independent)
 [UntrainableMetaTransform](cpp_api/untrainablemetatransform/untrainablemetatransform.md) | [UntrainableTransform](cpp_api/untrainabletransform/untrainabletransform.md) | A [Transform](cpp_api/transform/transform.md) that is *not* [independent](cpp_api/transform/members.md#independent) and cannot be trained
 [MetadataTransform](cpp_api/metadatatransform/metadatatransform.md) | [Transform](cpp_api/transform/transform.md) | A [Transform](cpp_api/transform/transform.md) that operates only on [Transform](cpp_api/transform/transform.md) [metadata](cpp_api/transform/members.md#file)
 [UntrainableMetadataTransform](cpp_api/untrainablemetadatatransform/untrainablemetadatatransform.md) | [MetadataTransform](cpp_api/metadatatransform/metadatatransform.md) | A [MetadataTransform](cpp_api/metadatatransform/metadatatransform.md) that cannot be trained
 [TimeVaryingTransform](cpp_api/timevaryingtransform/timevaryingtransform.md) | [Transform](cpp_api/transform/transform.md) | A [Transform](cpp_api/transform/transform.md) that changes at runtime as a result of the input
 [UntrainableDistance](cpp_api/untrainabledistance/untrainabledistance.md) | [Distance](cpp_api/distance/distance.md) | A [Distance](cpp_api/distance/distance.md) that cannot be trained
 [FileGallery](cpp_api/filegallery/filegallery.md) | [Gallery](cpp_api/gallery/gallery.md) | DOCUMENT ME
 [MatrixOutput](cpp_api/matrixoutput/matrixoutput.md) | [Output](cpp_api/output/output.md) | A [Output](cpp_api/output/output.md) that outputs data as a matrix
 
 As was stated before, all plugins in OpenBR are constructed using strings stored as [Files](cpp_api/file/file.md). The construction is done at runtime by a [Factory](cpp_api/factory/factory.md) class. The [Factory](cpp_api/factory/factory.md) expects strings of the form "PluginName(property1=value1,propert2=value2...propertyN=valueN)". It then looks up the "PluginName" in a static registry and builds the plugin if found. The registry is populated using a special macro [BR_REGISTER](cpp_api/factory/macros.md#br_register); each plugin needs to register itself and its base abstraction in the factory to enable construction. The purpose of this is to allow algorithms to be described completely by strings. For more information on algorithms in OpenBR please see the [tutorial](../tutorials.md#algorithms-in-openbr)
 