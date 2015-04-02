# Plugins

A description of all of the plugins available in OpenBR, broken down by module. This section assumes knowledge of the [C++ Plugin API](technical.md#c++ plugin api) and the [plugin abstractions](abstractions.md).

## Classification

---

#### AdaBoostTransform

Wraps OpenCV's Ada Boost framework

* **file:** classification/adaboost.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **author:** Scott Klum
* **properties:** None

---

#### EBIFTransform

Face Recognition Using Early Biologically Inspired Features

* **file:** classification/ebif.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### ForestTransform

Wraps OpenCV's random trees framework

* **file:** classification/forest.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **author:** Scott Klum
* **properties:** None

---

#### ForestInductionTransform

Wraps OpenCV's random trees framework to induce features

* **file:** classification/forest.cpp
* **inherits:** [ForestTransform](abstractions.md#foresttransform)
* **author:** Scott Klum
* **properties:** None

---

#### IPC2013Initializer

Initializes Intel Perceptual Computing SDK 2013

* **file:** classification/ipc2013.cpp
* **inherits:** [Initializer](abstractions.md#initializer)
* **author:** Josh Klontz
* **properties:** None

---

#### IPC2013FaceRecognitionTransfrom

Intel Perceptual Computing SDK 2013 Face Recognition

* **file:** classification/ipc2013.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### EigenInitializer

Initialize Eigen

* **file:** classification/lda.cpp
* **inherits:** [Initializer](abstractions.md#initializer)
* **author:** Scott Klum
* **properties:** None

---

#### PCATransform

Projects input into learned Principal Component Analysis subspace.

* **file:** classification/lda.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **authors:** Brendan Klare, Josh Klontz
* **properties:** None

---

#### RowWisePCATransform

PCA on each row.

* **file:** classification/lda.cpp
* **inherits:** [PCATransform](abstractions.md#pcatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### DFFSTransform

Computes Distance From Feature Space (DFFS)

* **file:** classification/lda.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **author:** Josh Klontz
* **properties:** None

---

#### LDATransform

Projects input into learned Linear Discriminant Analysis subspace.

* **file:** classification/lda.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **authors:** Brendan Klare, Josh Klontz
* **properties:** None

---

#### SparseLDATransform

Projects input into learned Linear Discriminant Analysis subspace

* **file:** classification/lda.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **author:** Brendan Klare
* **properties:** None

---

#### MLPTransform

Wraps OpenCV's multi-layer perceptron framework

* **file:** classification/mlp.cpp
* **inherits:** [MetaTransform](abstractions.md#metatransform)
* **author:** Scott Klum
* **properties:** None

---

#### NT4Initializer

Initialize Neurotech SDK 4

* **file:** classification/nt4.cpp
* **inherits:** [Initializer](abstractions.md#initializer)
* **authors:** Josh Klontz, E. Taborsky
* **properties:** None

---

#### NT4DetectFace

Neurotech face detection

* **file:** classification/nt4.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **authors:** Josh Klontz, E. Taborsky
* **properties:** None

---

#### NT4EnrollFace

Enroll face in Neurotech SDK 4

* **file:** classification/nt4.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### NT4EnrollIris

Enroll iris in Neurotech SDK 4

* **file:** classification/nt4.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** E. Taborsky
* **properties:** None

---

#### NT4Compare

Compare templates with Neurotech SDK 4

* **file:** classification/nt4.cpp
* **inherits:** [Distance](abstractions.md#distance)
* **authors:** Josh Klontz, E. Taborsky
* **properties:** None

---

#### PP4Initializer

Initialize PittPatt 4

* **file:** classification/pp4.cpp
* **inherits:** [Initializer](abstractions.md#initializer)
* **author:** Josh Klontz
* **properties:** None

---

#### PP4EnrollTransform

Enroll faces in PittPatt 4

* **file:** classification/pp4.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### PP5Initializer

Initialize PP5

* **file:** classification/pp5.cpp
* **inherits:** [Initializer](abstractions.md#initializer)
* **authors:** Josh Klontz, E. Taborsky
* **properties:** None

---

#### PP5EnrollTransform

Enroll faces in PP5

* **file:** classification/pp5.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **authors:** Josh Klontz, E. Taborsky
* **properties:** None

---

#### PP5CompareDistance

Compare templates with PP5

* **file:** classification/pp5.cpp
* **inherits:** [UntrainableDistance](abstractions.md#untrainabledistance)
* **authors:** Josh Klontz, E. Taborsky
* **properties:** None

---

#### SVMTransform

C. Burges. "A tutorial on support vector machines for pattern recognition,"

* **file:** classification/svm.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **author:** Josh Klontz
* **properties:** None

---

#### SVMDistance

SVM Regression on template absolute differences.

* **file:** classification/svm.cpp
* **inherits:** [Distance](abstractions.md#distance)
* **author:** Josh Klontz
* **properties:** None

---

#### TurkClassifierTransform

Convenience class for training turk attribute regressors

* **file:** classification/turk.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **author:** Josh Klontz
* **properties:** None

## Cluster

---

#### CollectNNTransform

Collect nearest neighbors and append them to metadata.

* **file:** cluster/collectnn.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Charles Otto
* **properties:** None

---

#### KMeansTransform

Wraps OpenCV kmeans and flann.

* **file:** cluster/kmeans.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **author:** Josh Klontz
* **properties:** None

---

#### KNNTransform

K nearest neighbors classifier.

* **file:** cluster/knn.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **author:** Josh Klontz
* **properties:** None

---

#### LogNNTransform

Log nearest neighbors to specified file.

* **file:** cluster/lognn.cpp
* **inherits:** [TimeVaryingTransform](abstractions.md#timevaryingtransform)
* **author:** Charles Otto
* **properties:** None

---

#### RandomCentroidsTransform

Chooses k random points to be centroids.

* **file:** cluster/randomcentroids.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **author:** Austin Blanton
* **properties:** None

## Core

---

#### AlgorithmsInitializer

Initializes global abbreviations with implemented algorithms

* **file:** core/algorithms.cpp
* **inherits:** [Initializer](abstractions.md#initializer)
* **author:** Josh Klontz
* **properties:** None

---

#### ProcrustesAlignTransform

Improved procrustes alignment of points, to include a post processing scaling of points

* **file:** core/align.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **author:** Brendan Klare
* **properties:** None

---

#### TextureMapTransform

Maps texture from one set of points to another. Assumes that points are rigidly transformed

* **file:** core/align.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **authors:** Brendan Klare, Scott Klum
* **properties:** None

---

#### SynthesizePointsTransform

Synthesize additional points via triangulation.

* **file:** core/align.cpp
* **inherits:** [MetadataTransform](abstractions.md#metadatatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### ProcrustesInitializer

Initialize Procrustes croppings

* **file:** core/align.cpp
* **inherits:** [Initializer](abstractions.md#initializer)
* **author:** Brendan Klare
* **properties:** None

---

#### AttributeAlgorithmsInitializer

Initializes global abbreviations with implemented algorithms for attributes

* **file:** core/attributealgorithms.cpp
* **inherits:** [Initializer](abstractions.md#initializer)
* **author:** Babatunde Ogunfemi
* **properties:** None

---

#### CacheTransform

Caches br::Transform::project() results.

* **file:** core/cache.cpp
* **inherits:** [MetaTransform](abstractions.md#metatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### ContractTransform

It's like the opposite of ExpandTransform, but not really

* **file:** core/contract.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Charles Otto
* **properties:** None

---

#### CrossValidateTransform

Cross validate a trainable transform.

* **file:** core/crossvalidate.cpp
* **inherits:** [MetaTransform](abstractions.md#metatransform)
* **authors:** Josh Klontz, Scott Klum
* **properties:** None

---

#### DiscardTransform

Removes all template's matrices.

* **file:** core/discard.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### ExpandTransform

Performs an expansion step on input templatelists

* **file:** core/expand.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### FirstTransform

Removes all but the first matrix from the template.

* **file:** core/first.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### ForkTransform

Transforms in parallel.

* **file:** core/fork.cpp
* **inherits:** [CompositeTransform](abstractions.md#compositetransform)
* **author:** Josh Klontz
* **properties:** None

---

#### FTETransform

Flags images that failed to enroll based on the specified transform.

* **file:** core/fte.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **author:** Josh Klontz
* **properties:** None

---

#### GalleryCompareTransform

Compare each template to a fixed gallery (with name = galleryName), using the specified distance.

* **file:** core/gallerycompare.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **author:** Charles Otto
* **properties:** None

---

#### IdentityTransform

A no-op transform.

* **file:** core/identity.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### IndependentTransform

Clones the transform so that it can be applied independently.

* **file:** core/independent.cpp
* **inherits:** [MetaTransform](abstractions.md#metatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### JNIInitializer

Initialize JNI

* **file:** core/jni.cpp
* **inherits:** [Initializer](abstractions.md#initializer)
* **author:** Jordan Cheney
* **properties:** None

---

#### LikelyTransform

Generic interface to Likely JIT compiler

* **file:** core/likely.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### LoadStoreTransform

Caches transform training.

* **file:** core/loadstore.cpp
* **inherits:** [MetaTransform](abstractions.md#metatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### PipeTransform

Transforms in series.

* **file:** core/pipe.cpp
* **inherits:** [CompositeTransform](abstractions.md#compositetransform)
* **author:** Josh Klontz
* **properties:** None

---

#### ProcessWrapperTransform

Interface to a separate process

* **file:** core/processwrapper.cpp
* **inherits:** [WrapperTransform](abstractions.md#wrappertransform)
* **author:** Charles Otto
* **properties:** None

---

#### Registrar

Register custom objects with Qt meta object system.

* **file:** core/registrar.cpp
* **inherits:** [Initializer](abstractions.md#initializer)
* **author:** Charles Otto
* **properties:** None

---

#### RestTransform

Removes the first matrix from the template.

* **file:** core/rest.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### SchrodingerTransform

Generates two templates, one of which is passed through a transform and the other

* **file:** core/schrodinger.cpp
* **inherits:** [MetaTransform](abstractions.md#metatransform)
* **author:** Scott Klum
* **properties:** None

---

#### SingletonTransform

A globally shared transform.

* **file:** core/singleton.cpp
* **inherits:** [MetaTransform](abstractions.md#metatransform)
* **author:** Josh Klontz
* **properties:** None

## Distance

---

#### AttributeDistance

Attenuation function based distance from attributes

* **file:** distance/attribute.cpp
* **inherits:** [UntrainableDistance](abstractions.md#untrainabledistance)
* **author:** Scott Klum
* **properties:** None

---

#### BayesianQuantizationDistance

Bayesian quantization distance

* **file:** distance/bayesianquantization.cpp
* **inherits:** [Distance](abstractions.md#distance)
* **author:** Josh Klontz
* **properties:** None

---

#### ByteL1Distance

Fast 8-bit L1 distance

* **file:** distance/byteL1.cpp
* **inherits:** [UntrainableDistance](abstractions.md#untrainabledistance)
* **author:** Josh Klontz
* **properties:** None

---

#### CrossValidateDistance

Cross validate a distance metric.

* **file:** distance/crossvalidate.cpp
* **inherits:** [UntrainableDistance](abstractions.md#untrainabledistance)
* **author:** Josh Klontz
* **properties:** None

---

#### DefaultDistance

DistDistance wrapper.

* **file:** distance/default.cpp
* **inherits:** [UntrainableDistance](abstractions.md#untrainabledistance)
* **author:** Josh Klontz
* **properties:** None

---

#### DistDistance

Standard distance metrics

* **file:** distance/dist.cpp
* **inherits:** [UntrainableDistance](abstractions.md#untrainabledistance)
* **author:** Josh Klontz
* **properties:** None

---

#### FilterDistance

Checks target metadata against filters.

* **file:** distance/filter.cpp
* **inherits:** [UntrainableDistance](abstractions.md#untrainabledistance)
* **author:** Josh Klontz
* **properties:** None

---

#### FuseDistance

Fuses similarity scores across multiple matrices of compared templates

* **file:** distance/fuse.cpp
* **inherits:** [Distance](abstractions.md#distance)
* **author:** Scott Klum
* **properties:** None

---

#### HalfByteL1Distance

Fast 4-bit L1 distance

* **file:** distance/halfbyteL1.cpp
* **inherits:** [UntrainableDistance](abstractions.md#untrainabledistance)
* **author:** Josh Klontz
* **properties:** None

---

#### HeatMapDistance

1v1 heat map comparison

* **file:** distance/heatmap.cpp
* **inherits:** [Distance](abstractions.md#distance)
* **author:** Scott Klum
* **properties:** None

---

#### IdenticalDistance

Returns

* **file:** distance/identical.cpp
* **inherits:** [UntrainableDistance](abstractions.md#untrainabledistance)
* **author:** Josh Klontz
* **properties:** None

---

#### KeyPointMatcherDistance

Wraps OpenCV Key Point Matcher

* **file:** distance/keypointmatcher.cpp
* **inherits:** [UntrainableDistance](abstractions.md#untrainabledistance)
* **author:** Josh Klontz
* **properties:** None

---

#### L1Distance

L1 distance computed using eigen.

* **file:** distance/L1.cpp
* **inherits:** [UntrainableDistance](abstractions.md#untrainabledistance)
* **author:** Josh Klontz
* **properties:** None

---

#### L2Distance

L2 distance computed using eigen.

* **file:** distance/L2.cpp
* **inherits:** [UntrainableDistance](abstractions.md#untrainabledistance)
* **author:** Josh Klontz
* **properties:** None

---

#### MatchProbabilityDistance

Match Probability

* **file:** distance/matchprobability.cpp
* **inherits:** [Distance](abstractions.md#distance)
* **author:** Josh Klontz
* **properties:** None

---

#### MetadataDistance

Checks target metadata against query metadata.

* **file:** distance/metadata.cpp
* **inherits:** [UntrainableDistance](abstractions.md#untrainabledistance)
* **author:** Scott Klum
* **properties:** None

---

#### NegativeLogPlusOneDistance

Returns -log(distance(a,b)+1)

* **file:** distance/neglogplusone.cpp
* **inherits:** [UntrainableDistance](abstractions.md#untrainabledistance)
* **author:** Josh Klontz
* **properties:** None

---

#### OnlineDistance

Online distance metric to attenuate match scores across multiple frames

* **file:** distance/online.cpp
* **inherits:** [UntrainableDistance](abstractions.md#untrainabledistance)
* **author:** Brendan klare
* **properties:** None

---

#### PipeDistance

Distances in series.

* **file:** distance/pipe.cpp
* **inherits:** [Distance](abstractions.md#distance)
* **author:** Josh Klontz
* **properties:** None

---

#### RejectDistance

Sets distance to -FLOAT_MAX if a target template has/doesn't have a key.

* **file:** distance/reject.cpp
* **inherits:** [UntrainableDistance](abstractions.md#untrainabledistance)
* **author:** Scott Klum
* **properties:** None

---

#### SumDistance

Sum match scores across multiple distances

* **file:** distance/sum.cpp
* **inherits:** [UntrainableDistance](abstractions.md#untrainabledistance)
* **author:** Scott Klum
* **properties:** None

---

#### TurkDistance

Unmaps Turk HITs to be compared against query mats

* **file:** distance/turk.cpp
* **inherits:** [UntrainableDistance](abstractions.md#untrainabledistance)
* **author:** Scott Klum
* **properties:** None

---

#### UnitDistance

Linear normalizes of a distance so the mean impostor score is 0 and the mean genuine score is 1.

* **file:** distance/unit.cpp
* **inherits:** [Distance](abstractions.md#distance)
* **author:** Josh Klontz
* **properties:** None

## Format

---

#### binaryFormat

A simple binary matrix format.

* **file:** format/binary.cpp
* **inherits:** [Format](abstractions.md#format)
* **author:** Josh Klontz
* **properties:** None

---

#### csvFormat

Reads a comma separated value file.

* **file:** format/csv.cpp
* **inherits:** [Format](abstractions.md#format)
* **author:** Josh Klontz
* **properties:** None

---

#### ebtsFormat

Reads FBI EBTS transactions.

* **file:** format/ebts.cpp
* **inherits:** [Format](abstractions.md#format)
* **author:** Scott Klum
* **properties:** None

---

#### lffsFormat

Reads a NIST LFFS file.

* **file:** format/lffs.cpp
* **inherits:** [Format](abstractions.md#format)
* **author:** Josh Klontz
* **properties:** None

---

#### lmatFormat

Likely matrix format

* **file:** format/lmat.cpp
* **inherits:** [Format](abstractions.md#format)
* **author:** Josh Klontz
* **properties:** None

---

#### matFormat

MATLAB <tt>.mat</tt> format.

* **file:** format/mat.cpp
* **inherits:** [Format](abstractions.md#format)
* **author:** Josh Klontz
* **properties:** None

---

#### mtxFormat

Reads a NIST BEE similarity matrix.

* **file:** format/mtx.cpp
* **inherits:** [Format](abstractions.md#format)
* **author:** Josh Klontz
* **properties:** None

---

#### maskFormat

Reads a NIST BEE mask matrix.

* **file:** format/mtx.cpp
* **inherits:** [mtxFormat](abstractions.md#mtxformat)
* **author:** Josh Klontz
* **properties:** None

---

#### nullFormat

Returns an empty matrix.

* **file:** format/null.cpp
* **inherits:** [Format](abstractions.md#format)
* **author:** Josh Klontz
* **properties:** None

---

#### postFormat

Handle POST requests

* **file:** format/post.cpp
* **inherits:** [Format](abstractions.md#format)
* **author:** Josh Klontz
* **properties:** None

---

#### rawFormat

RAW format

* **file:** format/raw.cpp
* **inherits:** [Format](abstractions.md#format)
* **author:** Josh Klontz
* **properties:** None

---

#### scoresFormat

Reads in scores or ground truth from a text table.

* **file:** format/scores.cpp
* **inherits:** [Format](abstractions.md#format)
* **author:** Josh Klontz
* **properties:** None

---

#### urlFormat

Reads image files from the web.

* **file:** format/url.cpp
* **inherits:** [Format](abstractions.md#format)
* **author:** Josh Klontz
* **properties:** None

---

#### videoFormat

Read all frames of a video using OpenCV

* **file:** format/video.cpp
* **inherits:** [Format](abstractions.md#format)
* **author:** Charles Otto
* **properties:** None

---

#### webcamFormat

Retrieves an image from a webcam.

* **file:** format/video.cpp
* **inherits:** [Format](abstractions.md#format)
* **author:** Josh Klontz
* **properties:** None

---

#### DefaultFormat

Reads image files.

* **file:** format/video.cpp
* **inherits:** [Format](abstractions.md#format)
* **author:** Josh Klontz
* **properties:** None

---

#### xmlFormat

Decodes images from Base64 xml

* **file:** format/xml.cpp
* **inherits:** [Format](abstractions.md#format)
* **authors:** Scott Klum, Josh Klontz
* **properties:** None

## Gallery

---

#### arffGallery

Weka ARFF file format.

* **file:** gallery/arff.cpp
* **inherits:** [Gallery](abstractions.md#gallery)
* **author:** Josh Klontz
* **properties:** None

---

#### galGallery

A binary gallery.

* **file:** gallery/binary.cpp
* **inherits:** [BinaryGallery](abstractions.md#binarygallery)
* **author:** Josh Klontz
* **properties:** None

---

#### utGallery

A contiguous array of br_universal_template.

* **file:** gallery/binary.cpp
* **inherits:** [BinaryGallery](abstractions.md#binarygallery)
* **author:** Josh Klontz
* **properties:** None

---

#### urlGallery

Newline-separated URLs.

* **file:** gallery/binary.cpp
* **inherits:** [BinaryGallery](abstractions.md#binarygallery)
* **author:** Josh Klontz
* **properties:** None

---

#### jsonGallery

Newline-separated JSON objects.

* **file:** gallery/binary.cpp
* **inherits:** [BinaryGallery](abstractions.md#binarygallery)
* **author:** Josh Klontz
* **properties:** None

---

#### crawlGallery

Crawl a root location for image files.

* **file:** gallery/crawl.cpp
* **inherits:** [Gallery](abstractions.md#gallery)
* **author:** Josh Klontz
* **properties:** None

---

#### csvGallery

Treats each line as a file.

* **file:** gallery/csv.cpp
* **inherits:** [FileGallery](abstractions.md#filegallery)
* **author:** Josh Klontz
* **properties:** None

---

#### dbGallery

Database input.

* **file:** gallery/db.cpp
* **inherits:** [Gallery](abstractions.md#gallery)
* **author:** Josh Klontz
* **properties:** None

---

#### DefaultGallery

Treats the gallery as a br::Format.

* **file:** gallery/default.cpp
* **inherits:** [Gallery](abstractions.md#gallery)
* **author:** Josh Klontz
* **properties:** None

---

#### EmptyGallery

Reads/writes templates to/from folders.

* **file:** gallery/empty.cpp
* **inherits:** [Gallery](abstractions.md#gallery)
* **author:** Josh Klontz
* **properties:** None

---

#### FDDBGallery

Implements the FDDB detection format.

* **file:** gallery/fddb.cpp
* **inherits:** [Gallery](abstractions.md#gallery)
* **author:** Josh Klontz
* **properties:** None

---

#### flatGallery

Treats each line as a call to File::flat()

* **file:** gallery/flat.cpp
* **inherits:** [FileGallery](abstractions.md#filegallery)
* **author:** Josh Klontz
* **properties:** None

---

#### googleGallery

Input from a google image search.

* **file:** gallery/google.cpp
* **inherits:** [Gallery](abstractions.md#gallery)
* **author:** Josh Klontz
* **properties:** None

---

#### keyframesGallery

Read key frames of a video with LibAV

* **file:** gallery/keyframes.cpp
* **inherits:** [Gallery](abstractions.md#gallery)
* **author:** Ben Klein
* **properties:** None

---

#### mp4Gallery

Read key frames of a .mp4 video file with LibAV

* **file:** gallery/keyframes.cpp
* **inherits:** [keyframesGallery](abstractions.md#keyframesgallery)
* **author:** Ben Klein
* **properties:** None

---

#### landmarksGallery

Text format for associating anonymous landmarks with images.

* **file:** gallery/landmarks.cpp
* **inherits:** [Gallery](abstractions.md#gallery)
* **author:** Josh Klontz
* **properties:** None

---

#### lmatGallery

Likely matrix format

* **file:** gallery/lmat.cpp
* **inherits:** [Gallery](abstractions.md#gallery)
* **author:** Josh Klontz
* **properties:** None

---

#### matrixGallery

Combine all templates into one large matrix and process it as a br::Format

* **file:** gallery/matrix.cpp
* **inherits:** [Gallery](abstractions.md#gallery)
* **author:** Josh Klontz
* **properties:** None

---

#### MemoryGalleries

Initialization support for memGallery.

* **file:** gallery/mem.cpp
* **inherits:** [Initializer](abstractions.md#initializer)
* **author:** Josh Klontz
* **properties:** None

---

#### memGallery

A gallery held in memory.

* **file:** gallery/mem.cpp
* **inherits:** [Gallery](abstractions.md#gallery)
* **author:** Josh Klontz
* **properties:** None

---

#### postGallery

Handle POST requests

* **file:** gallery/post.cpp
* **inherits:** [Gallery](abstractions.md#gallery)
* **author:** Josh Klontz
* **properties:** None

---

#### statGallery

Print template statistics.

* **file:** gallery/stat.cpp
* **inherits:** [Gallery](abstractions.md#gallery)
* **author:** Josh Klontz
* **properties:** None

---

#### templateGallery

Treat the file as a single binary template.

* **file:** gallery/template.cpp
* **inherits:** [Gallery](abstractions.md#gallery)
* **author:** Josh Klontz
* **properties:** None

---

#### turkGallery

For Amazon Mechanical Turk datasets

* **file:** gallery/turk.cpp
* **inherits:** [Gallery](abstractions.md#gallery)
* **author:** Scott Klum
* **properties:** None

---

#### txtGallery

Treats each line as a file.

* **file:** gallery/txt.cpp
* **inherits:** [FileGallery](abstractions.md#filegallery)
* **author:** Josh Klontz
* **properties:** None

---

#### xmlGallery

A

* **file:** gallery/xml.cpp
* **inherits:** [FileGallery](abstractions.md#filegallery)
* **author:** Josh Klontz
* **properties:** None

## Gui

---

#### AdjacentOverlayTransform

Load the image named in the specified property, draw it on the current matrix adjacent to the rect specified in the other property.

* **file:** gui/adjacentoverlay.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **author:** Charles Otto
* **properties:** None

---

#### DrawTransform

Renders metadata onto the image.

* **file:** gui/draw.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### DrawDelaunayTransform

Creates a Delaunay triangulation based on a set of points

* **file:** gui/drawdelaunay.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Scott Klum
* **properties:** None

---

#### DrawGridLinesTransform

Draws a grid on the image

* **file:** gui/drawgridlines.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### DrawOpticalFlow

Draw a line representing the direction and magnitude of optical flow at the specified points.

* **file:** gui/drawopticalflow.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Austin Blanton
* **properties:** None

---

#### DrawPropertiesPointTransform

Draw the values of a list of properties at the specified point on the image

* **file:** gui/drawpropertiespoint.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Charles Otto
* **properties:** None

---

#### DrawPropertyPointTransform

Draw the value of the specified property at the specified point on the image

* **file:** gui/drawpropertypoint.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Charles Otto
* **properties:** None

---

#### DrawSegmentation

Fill in the segmentations or draw a line between intersecting segments.

* **file:** gui/drawsegmentation.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Austin Blanton
* **properties:** None

---

#### ShowTransform

Displays templates in a GUI pop-up window using QT.

* **file:** gui/show.cpp
* **inherits:** [TimeVaryingTransform](abstractions.md#timevaryingtransform)
* **author:** Charles Otto
* **properties:** None

---

#### ShowTrainingTransform

Show the training data

* **file:** gui/show.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **author:** Josh Klontz
* **properties:** None

---

#### ManualTransform

Manual selection of landmark locations

* **file:** gui/show.cpp
* **inherits:** [ShowTransform](abstractions.md#showtransform)
* **author:** Scott Klum
* **properties:** None

---

#### ManualRectsTransform

Manual select rectangular regions on an image.

* **file:** gui/show.cpp
* **inherits:** [ShowTransform](abstractions.md#showtransform)
* **author:** Charles Otto
* **properties:** None

---

#### ElicitTransform

Elicits metadata for templates in a pretty GUI

* **file:** gui/show.cpp
* **inherits:** [ShowTransform](abstractions.md#showtransform)
* **author:** Scott Klum
* **properties:** None

---

#### SurveyTransform

Display an image, and asks a yes/no question about it

* **file:** gui/show.cpp
* **inherits:** [ShowTransform](abstractions.md#showtransform)
* **author:** Charles Otto
* **properties:** None

---

#### FPSLimit

Limits the frequency of projects going through this transform to the input targetFPS

* **file:** gui/show.cpp
* **inherits:** [TimeVaryingTransform](abstractions.md#timevaryingtransform)
* **author:** Charles Otto
* **properties:** None

---

#### FPSCalc

Calculates the average FPS of projects going through this transform, stores the result in AvgFPS

* **file:** gui/show.cpp
* **inherits:** [TimeVaryingTransform](abstractions.md#timevaryingtransform)
* **author:** Charles Otto
* **properties:** None

## Imgproc

---

#### AbsTransform

Computes the absolute value of each element.

* **file:** imgproc/abs.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### AbsDiffTransform

Take the absolute difference of two matrices.

* **file:** imgproc/absdiff.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### AdaptiveThresholdTransform

Wraps OpenCV's adaptive thresholding.

* **file:** imgproc/adaptivethreshold.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Scott Klum
* **properties:** None

---

#### AffineTransform

Performs a two or three point registration.

* **file:** imgproc/affine.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### AndTransform

Logical AND of two matrices.

* **file:** imgproc/and.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### ApplyMaskTransform

Applies a mask from the metadata.

* **file:** imgproc/applymask.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Austin Blanton
* **properties:** None

---

#### BayesianQuantizationTransform

Quantize into a space where L1 distance approximates log-likelihood.

* **file:** imgproc/bayesianquantization.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **author:** Josh Klontz
* **properties:** None

---

#### BinarizeTransform

Approximate floats as signed bit.

* **file:** imgproc/binarize.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### BlendTransform

Alpha-blend two matrices

* **file:** imgproc/blend.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### BlurTransform

Gaussian blur

* **file:** imgproc/blur.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### ByRowTransform

Turns each row into its own matrix.

* **file:** imgproc/byrow.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### CannyTransform

Warpper to OpenCV Canny edge detector

* **file:** imgproc/canny.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Scott Klum
* **properties:** None

---

#### CatTransform

Concatenates all input matrices into a single matrix.

* **file:** imgproc/cat.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### CatColsTransform

Concatenates all input matrices by column into a single matrix.

* **file:** imgproc/catcols.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Austin Blanton
* **properties:** None

---

#### CatRowsTransform

Concatenates all input matrices by row into a single matrix.

* **file:** imgproc/catrows.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### CenterTransform

Normalize each dimension based on training data.

* **file:** imgproc/center.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **author:** Josh Klontz
* **properties:** None

---

#### ContrastEqTransform

Xiaoyang Tan; Triggs, B.;

* **file:** imgproc/contrasteq.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### CropTransform

Crops about the specified region of interest.

* **file:** imgproc/crop.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### CropBlackTransform

Crop out black borders

* **file:** imgproc/cropblack.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### CropFromMaskTransform

Crops image based on mask metadata

* **file:** imgproc/cropfrommask.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Brendan Klare
* **properties:** None

---

#### CropSquareTransform

Trim the image so the width and the height are the same size.

* **file:** imgproc/cropsquare.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### CryptographicHashTransform

Wraps QCryptographicHash

* **file:** imgproc/cryptographichash.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### CvtTransform

Colorspace conversion.

* **file:** imgproc/cvt.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### CvtFloatTransform

Convert to floating point format.

* **file:** imgproc/cvtfloat.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### CvtUCharTransform

Convert to uchar format

* **file:** imgproc/cvtuchar.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### NLMeansDenoisingTransform

Wraps OpenCV Non-Local Means Denoising

* **file:** imgproc/denoising.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### DiscardAlphaTransform

Drop the alpha channel (if exists).

* **file:** imgproc/discardalpha.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Austin Blanton
* **properties:** None

---

#### DivTransform

Enforce a multiple of

* **file:** imgproc/div.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### DoGTransform

Difference of gaussians

* **file:** imgproc/dog.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### DownsampleTransform

Downsample the rows and columns of a matrix.

* **file:** imgproc/downsample.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Lacey Best-Rowden
* **properties:** None

---

#### DupTransform

Duplicates the template data.

* **file:** imgproc/dup.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### EnsureChannelsTransform

Enforce the matrix has a certain number of channels by adding or removing channels.

* **file:** imgproc/ensurechannels.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### EqualizeHistTransform

Histogram equalization

* **file:** imgproc/equalizehist.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### FlipTransform

Flips the image about an axis.

* **file:** imgproc/flip.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### FloodTransform

Fill black pixels with the specified color.

* **file:** imgproc/flood.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### GaborTransform

http://en.wikipedia.org/wiki/Gabor_filter

* **file:** imgproc/gabor.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### GaborJetTransform

A vector of gabor wavelets applied at a point.

* **file:** imgproc/gabor.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### GammaTransform

Gamma correction

* **file:** imgproc/gamma.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### GradientTransform

Computes magnitude and/or angle of image.

* **file:** imgproc/gradient.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### GradientMaskTransform

Masks image according to pixel change.

* **file:** imgproc/gradientmask.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### GroupTransform

Group all input matrices into a single matrix.

* **file:** imgproc/group.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### HistTransform

Histograms the matrix

* **file:** imgproc/hist.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### HistBinTransform

Quantizes the values into bins.

* **file:** imgproc/histbin.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### HistEqQuantizationTransform

Approximate floats as uchar with different scalings for each dimension.

* **file:** imgproc/histeqquantization.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **author:** Josh Klontz
* **properties:** None

---

#### HoGDescriptorTransform

OpenCV HOGDescriptor wrapper

* **file:** imgproc/hog.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Austin Blanton
* **properties:** None

---

#### InpaintTransform

Wraps OpenCV inpainting

* **file:** imgproc/inpaint.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### IntegralTransform

Computes integral image.

* **file:** imgproc/integral.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### IntegralHistTransform

An integral histogram

* **file:** imgproc/integralhist.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### IntegralSamplerTransform

Sliding window feature extraction from a multi-channel integral image.

* **file:** imgproc/integralsampler.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### KernelHashTransform

Kernel hash

* **file:** imgproc/kernelhash.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### KeyPointDescriptorTransform

Wraps OpenCV Key Point Descriptor

* **file:** imgproc/keypointdescriptor.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### LargestConvexAreaTransform

Set the template's label to the area of the largest convex hull.

* **file:** imgproc/largestconvexarea.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### LBPTransform

Ahonen, T.; Hadid, A.; Pietikainen, M.;

* **file:** imgproc/lbp.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### LimitSizeTransform

Limit the size of the template

* **file:** imgproc/limitsize.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### LTPTransform

Tan, Xiaoyang, and Bill Triggs. "Enhanced local texture feature sets for face recognition under difficult lighting conditions." Analysis and Modeling of Faces and Gestures. Springer Berlin Heidelberg, 2007. 168-182.

* **file:** imgproc/ltp.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **authors:** Brendan Klare, Josh Klontz
* **properties:** None

---

#### MAddTransform

dst = a*src+b

* **file:** imgproc/madd.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### MaskTransform

Applies an eliptical mask

* **file:** imgproc/mask.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### MatStatsTransform

Statistics

* **file:** imgproc/matstats.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### MeanTransform

Computes the mean of a set of templates.

* **file:** imgproc/mean.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **author:** Scott Klum
* **properties:** None

---

#### MeanFillTransform

Fill 0 pixels with the mean of non-0 pixels.

* **file:** imgproc/meanfill.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### MergeTransform

Wraps OpenCV merge

* **file:** imgproc/merge.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### MorphTransform

Morphological operator

* **file:** imgproc/morph.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### BuildScalesTransform

Document me

* **file:** imgproc/multiscale.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **author:** Austin Blanton
* **properties:** None

---

#### NormalizeTransform

Normalize matrix to unit length

* **file:** imgproc/normalize.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:**

	* **NormType**-  Values are NORM_INF, NORM_L1, NORM_L2, NORM_MINMAX
	* **ByRow**-  If true normalize each row independently otherwise normalize the entire matrix.
	* **alpha**-  Lower bound if using NORM_MINMAX. Value to normalize to otherwise.
	* **beta**-  Upper bound if using NORM_MINMAX. Not used otherwise.
	* **squareRoot**-  If true compute the signed square root of the output after normalization.


---

#### OrigLinearRegressionTransform

Prediction with magic numbers from jmp; must get input as blue;green;red

* **file:** imgproc/origlinearregression.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** E. Taborsky
* **properties:** None

---

#### PackTransform

Compress two uchar into one uchar.

* **file:** imgproc/pack.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### PowTransform

Raise each element to the specified power.

* **file:** imgproc/pow.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### ProductQuantizationDistance

Distance in a product quantized space

* **file:** imgproc/productquantization.cpp
* **inherits:** [UntrainableDistance](abstractions.md#untrainabledistance)
* **author:** Josh Klontz
* **properties:** None

---

#### ProductQuantizationTransform

Product quantization

* **file:** imgproc/productquantization.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **author:** Josh Klontz
* **properties:** None

---

#### QuantizeTransform

Approximate floats as uchar.

* **file:** imgproc/quantize.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **author:** Josh Klontz
* **properties:** None

---

#### RankTransform

Converts each element to its rank-ordered value.

* **file:** imgproc/rank.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### RectRegionsTransform

Subdivide matrix into rectangular subregions.

* **file:** imgproc/rectregions.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### RecursiveIntegralSamplerTransform

Construct template in a recursive decent manner.

* **file:** imgproc/recursiveintegralsampler.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **author:** Josh Klontz
* **properties:** None

---

#### RedLinearRegressionTransform

Prediction using only the red wavelength; magic numbers from jmp

* **file:** imgproc/redlinearregression.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** E. Taborsky
* **properties:** None

---

#### ReshapeTransform

Reshape the each matrix to the specified number of rows.

* **file:** imgproc/reshape.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### ResizeTransform

Resize the template

* **file:** imgproc/resize.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### RevertAffineTransform

Designed for use after eye detection + Stasm, this will

* **file:** imgproc/revertaffine.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Brendan Klare
* **properties:** None

---

#### RGTransform

Normalized RG color space.

* **file:** imgproc/rg.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### RndPointTransform

Generates a random landmark.

* **file:** imgproc/rndpoint.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **author:** Josh Klontz
* **properties:** None

---

#### RndRegionTransform

Selects a random region.

* **file:** imgproc/rndregion.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### RndRotateTransform

Randomly rotates an image in a specified range.

* **file:** imgproc/rndrotate.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Scott Klum
* **properties:** None

---

#### RndSubspaceTransform

Generates a random subspace.

* **file:** imgproc/rndsubspace.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **author:** Josh Klontz
* **properties:** None

---

#### ROITransform

Crops the rectangular regions of interest.

* **file:** imgproc/roi.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### ROIFromPtsTransform

Crops the rectangular regions of interest from given points and sizes.

* **file:** imgproc/roifrompoints.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Austin Blanton
* **properties:** None

---

#### RootNormTransform

dst=sqrt(norm_L1(src)) proposed as RootSIFT in

* **file:** imgproc/rootnorm.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### RowWiseMeanCenterTransform

Remove the row-wise training set average.

* **file:** imgproc/rowwisemeancenter.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **author:** Josh Klontz
* **properties:** None

---

#### SampleFromMaskTransform

Samples pixels from a mask.

* **file:** imgproc/samplefrommask.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Scott Klum
* **properties:** None

---

#### ScaleTransform

Scales using the given factor

* **file:** imgproc/scale.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Scott Klum
* **properties:** None

---

#### SIFTDescriptorTransform

Specialize wrapper OpenCV SIFT wrapper

* **file:** imgproc/sift.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### SkinMaskTransform

http://worldofcameras.wordpress.com/tag/skin-detection-opencv/

* **file:** imgproc/skinmask.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### SlidingWindowTransform

Applies a transform to a sliding window.

* **file:** imgproc/slidingwindow.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **author:** Austin Blanton
* **properties:** None

---

#### IntegralSlidingWindowTransform

Overloads SlidingWindowTransform for integral images that should be

* **file:** imgproc/slidingwindow.cpp
* **inherits:** [SlidingWindowTransform](abstractions.md#slidingwindowtransform)
* **author:** Josh Klontz
* **properties:** None

---

#### SplitChannelsTransform

Split a multi-channel matrix into several single-channel matrices.

* **file:** imgproc/splitchannels.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### SubdivideTransform

Divide the matrix into 4 smaller matricies of equal size.

* **file:** imgproc/subdivide.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### SubtractTransform

Subtract two matrices.

* **file:** imgproc/subtract.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### ThresholdTransform

Wraps OpenCV's adaptive thresholding.

* **file:** imgproc/threshold.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Scott Klum
* **properties:** None

---

#### WatershedSegmentationTransform

Applies watershed segmentation.

* **file:** imgproc/watershedsegmentation.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Austin Blanton
* **properties:** None

## Io

---

#### DecodeTransform

Decodes images

* **file:** io/decode.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### DownloadTransform

Downloads an image from a URL

* **file:** io/download.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### IncrementalOutputTransform

Incrementally output templates received to a gallery, based on the current filename

* **file:** io/incrementaloutput.cpp
* **inherits:** [TimeVaryingTransform](abstractions.md#timevaryingtransform)
* **author:** Charles Otto
* **properties:** None

---

#### OpenTransform

Applies br::Format to br::Template::file::name and appends results.

* **file:** io/open.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### PrintTransform

Prints the template's file to stdout or stderr.

* **file:** io/print.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### ReadTransform

Read images

* **file:** io/read.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### ReadLandmarksTransform

Read landmarks from a file and associate them with the correct templates.

* **file:** io/readlandmarks.cpp
* **inherits:** [UntrainableMetadataTransform](abstractions.md#untrainablemetadatatransform)
* **author:** Scott Klum
* **properties:** None

---

#### WriteTransform

Write all mats to disk as images.

* **file:** io/write.cpp
* **inherits:** [TimeVaryingTransform](abstractions.md#timevaryingtransform)
* **author:** Brendan Klare
* **properties:** None

---

#### YouTubeFacesDBTransform

Implements the YouTubesFaceDB

* **file:** io/youtubefacesdb.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

## Metadata

---

#### AnonymizeLandmarksTransform

Remove a name from a point/rect

* **file:** metadata/anonymizelandmarks.cpp
* **inherits:** [UntrainableMetadataTransform](abstractions.md#untrainablemetadatatransform)
* **author:** Scott Klum
* **properties:** None

---

#### AsTransform

Change the br::Template::file extension

* **file:** metadata/as.cpp
* **inherits:** [UntrainableMetadataTransform](abstractions.md#untrainablemetadatatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### AveragePointsTransform

Averages a set of landmarks into a new landmark

* **file:** metadata/averagepoints.cpp
* **inherits:** [UntrainableMetadataTransform](abstractions.md#untrainablemetadatatransform)
* **author:** Brendan Klare
* **properties:** None

---

#### CascadeTransform

Wraps OpenCV cascade classifier

* **file:** metadata/cascade.cpp
* **inherits:** [MetaTransform](abstractions.md#metatransform)
* **authors:** Josh Klontz, David Crouse
* **properties:** None

---

#### CheckTransform

Checks the template for NaN values.

* **file:** metadata/check.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### ClearPointsTransform

Clears the points from a template

* **file:** metadata/clearpoints.cpp
* **inherits:** [UntrainableMetadataTransform](abstractions.md#untrainablemetadatatransform)
* **author:** Brendan Klare
* **properties:** None

---

#### ConsolidateDetectionsTransform

Consolidate redundant/overlapping detections.

* **file:** metadata/consolidatedetections.cpp
* **inherits:** [UntrainableMetadataTransform](abstractions.md#untrainablemetadatatransform)
* **author:** Brendan Klare
* **properties:** None

---

#### CropRectTransform

Crops the width and height of a template's rects by input width and height factors.

* **file:** metadata/croprect.cpp
* **inherits:** [UntrainableMetadataTransform](abstractions.md#untrainablemetadatatransform)
* **author:** Scott Klum
* **properties:** None

---

#### DelaunayTransform

Creates a Delaunay triangulation based on a set of points

* **file:** metadata/delaunay.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Scott Klum
* **properties:** None

---

#### ExpandRectTransform

Expand the width and height of a template's rects by input width and height factors.

* **file:** metadata/expandrect.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Charles Otto
* **properties:** None

---

#### ExtractMetadataTransform

Create matrix from metadata values.

* **file:** metadata/extractmetadata.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### ASEFEyesTransform

Bolme, D.S.; Draper, B.A.; Beveridge, J.R.;

* **file:** metadata/eyes.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **authors:** David Bolme, Josh Klontz
* **properties:** None

---

#### FilterDupeMetadataTransform

Removes duplicate templates based on a unique metadata key

* **file:** metadata/filterdupemetadata.cpp
* **inherits:** [TimeVaryingTransform](abstractions.md#timevaryingtransform)
* **author:** Austin Blanton
* **properties:** None

---

#### GridTransform

Add landmarks to the template in a grid layout

* **file:** metadata/grid.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### GroundTruthTransform

Add any ground truth to the template using the file's base name.

* **file:** metadata/groundtruth.cpp
* **inherits:** [UntrainableMetadataTransform](abstractions.md#untrainablemetadatatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### HOGPersonDetectorTransform

Detects objects with OpenCV's built-in HOG detection.

* **file:** metadata/hogpersondetector.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Austin Blanton
* **properties:** None

---

#### IfMetadataTransform

Clear templates without the required metadata.

* **file:** metadata/ifmetadata.cpp
* **inherits:** [UntrainableMetadataTransform](abstractions.md#untrainablemetadatatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### ImpostorUniquenessMeasureTransform

Impostor Uniqueness Measure

* **file:** metadata/imposteruniquenessmeasure.cpp
* **inherits:** [Transform](abstractions.md#transform)
* **author:** Josh Klontz
* **properties:** None

---

#### JSONTransform

Represent the metadata as JSON template data.

* **file:** metadata/json.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### KeepMetadataTransform

Retains only the values for the keys listed, to reduce template size

* **file:** metadata/keepmetadata.cpp
* **inherits:** [UntrainableMetadataTransform](abstractions.md#untrainablemetadatatransform)
* **author:** Scott Klum
* **properties:** None

---

#### KeyPointDetectorTransform

Wraps OpenCV Key Point Detector

* **file:** metadata/keypointdetector.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

#### KeyToRectTransform

Convert values of key_X, key_Y, key_Width, key_Height to a rect.

* **file:** metadata/keytorect.cpp
* **inherits:** [UntrainableMetadataTransform](abstractions.md#untrainablemetadatatransform)
* **author:** Jordan Cheney
* **properties:** None

---

#### MongooseInitializer

Initialize mongoose server

* **file:** metadata/mongoose.cpp
* **inherits:** [Initializer](abstractions.md#initializer)
* **author:** Unknown
* **properties:** None

---

#### NameTransform

Sets the template's matrix data to the br::File::name.

* **file:** metadata/name.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### NameLandmarksTransform

Name a point/rect

* **file:** metadata/namelandmarks.cpp
* **inherits:** [UntrainableMetadataTransform](abstractions.md#untrainablemetadatatransform)
* **author:** Scott Klum
* **properties:** None

---

#### NormalizePointsTransform

Normalize points to be relative to a single point

* **file:** metadata/normalizepoints.cpp
* **inherits:** [UntrainableMetadataTransform](abstractions.md#untrainablemetadatatransform)
* **author:** Scott Klum
* **properties:** None

---

#### PointDisplacementTransform

Normalize points to be relative to a single point

* **file:** metadata/pointdisplacement.cpp
* **inherits:** [UntrainableMetadataTransform](abstractions.md#untrainablemetadatatransform)
* **author:** Scott Klum
* **properties:** None

---

#### PointsToMatrixTransform

Converts either the file::points() list or a QList<QPointF> metadata item to be the template's matrix

* **file:** metadata/pointstomatrix.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Scott Klum
* **properties:** None

---

#### ProcrustesTransform

Procrustes alignment of points

* **file:** metadata/procrustes.cpp
* **inherits:** [MetadataTransform](abstractions.md#metadatatransform)
* **author:** Scott Klum
* **properties:** None

---

#### RectsToTemplatesTransform

For each rectangle bounding box in src, a new

* **file:** metadata/rectstotemplates.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Brendan Klare
* **properties:** None

---

#### RegexPropertyTransform

Apply the input regular expression to the value of inputProperty, store the matched portion in outputProperty.

* **file:** metadata/regexproperty.cpp
* **inherits:** [UntrainableMetadataTransform](abstractions.md#untrainablemetadatatransform)
* **author:** Charles Otto
* **properties:** None

---

#### RemoveMetadataTransform

Removes a metadata field from all templates

* **file:** metadata/removemetadata.cpp
* **inherits:** [UntrainableMetadataTransform](abstractions.md#untrainablemetadatatransform)
* **author:** Brendan Klare
* **properties:** None

---

#### RemoveTemplatesTransform

Remove templates with the specified file extension or metadata value.

* **file:** metadata/removetemplates.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### RenameTransform

Rename metadata key

* **file:** metadata/rename.cpp
* **inherits:** [UntrainableMetadataTransform](abstractions.md#untrainablemetadatatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### RenameFirstTransform

Rename first found metadata key

* **file:** metadata/renamefirst.cpp
* **inherits:** [UntrainableMetadataTransform](abstractions.md#untrainablemetadatatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### ReorderPointsTransform

Reorder the points such that points[from[i]] becomes points[to[i]] and

* **file:** metadata/reorderpoints.cpp
* **inherits:** [UntrainableMetadataTransform](abstractions.md#untrainablemetadatatransform)
* **author:** Scott Klum
* **properties:** None

---

#### RestoreMatTransform

Set the last matrix of the input template to a matrix stored as metadata with input propName.

* **file:** metadata/restoremat.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Charles Otto
* **properties:** None

---

#### SaveMatTransform

Store the last matrix of the input template as a metadata key with input property name.

* **file:** metadata/savemat.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Charles Otto
* **properties:** None

---

#### SelectPointsTransform

Retains only landmarks/points at the provided indices

* **file:** metadata/selectpoints.cpp
* **inherits:** [UntrainableMetadataTransform](abstractions.md#untrainablemetadatatransform)
* **author:** Brendan Klare
* **properties:** None

---

#### SetMetadataTransform

Sets the metadata key/value pair.

* **file:** metadata/setmetadata.cpp
* **inherits:** [UntrainableMetadataTransform](abstractions.md#untrainablemetadatatransform)
* **author:** Josh Klontz
* **properties:** None

---

#### SetPointsInRectTransform

Set points relative to a rect

* **file:** metadata/setpointsinrect.cpp
* **inherits:** [UntrainableMetadataTransform](abstractions.md#untrainablemetadatatransform)
* **author:** Jordan Cheney
* **properties:** None

---

#### StasmInitializer

Initialize Stasm

* **file:** metadata/stasm4.cpp
* **inherits:** [Initializer](abstractions.md#initializer)
* **author:** Scott Klum
* **properties:** None

---

#### StasmTransform

Wraps STASM key point detector

* **file:** metadata/stasm4.cpp
* **inherits:** [UntrainableTransform](abstractions.md#untrainabletransform)
* **author:** Scott Klum
* **properties:** None

---

#### StopWatchTransform

Gives time elapsed over a specified transform as a function of both images (or frames) and pixels.

* **file:** metadata/stopwatch.cpp
* **inherits:** [MetaTransform](abstractions.md#metatransform)
* **authors:** Jordan Cheney, Josh Klontz
* **properties:** None

## Output

---

#### bestOutput

The highest scoring matches.

* **file:** output/best.cpp
* **inherits:** [Output](abstractions.md#output)
* **author:** Josh Klontz
* **properties:** None

---

#### csvOutput

Comma separated values output.

* **file:** output/csv.cpp
* **inherits:** [MatrixOutput](abstractions.md#matrixoutput)
* **author:** Josh Klontz
* **properties:** None

---

#### DefaultOutput

Adaptor class -- write a matrix output using Format classes.

* **file:** output/default.cpp
* **inherits:** [MatrixOutput](abstractions.md#matrixoutput)
* **author:** Charles Otto
* **properties:** None

---

#### EmptyOutput

Output to the terminal.

* **file:** output/empty.cpp
* **inherits:** [MatrixOutput](abstractions.md#matrixoutput)
* **author:** Josh Klontz
* **properties:** None

---

#### evalOutput

Evaluate the output matrix.

* **file:** output/eval.cpp
* **inherits:** [MatrixOutput](abstractions.md#matrixoutput)
* **author:** Josh Klontz
* **properties:** None

---

#### heatOutput

Matrix-like output for heat maps.

* **file:** output/heat.cpp
* **inherits:** [MatrixOutput](abstractions.md#matrixoutput)
* **author:** Scott Klum
* **properties:** None

---

#### histOutput

Score histogram.

* **file:** output/hist.cpp
* **inherits:** [Output](abstractions.md#output)
* **author:** Josh Klontz
* **properties:** None

---

#### meltOutput

One score per row.

* **file:** output/melt.cpp
* **inherits:** [MatrixOutput](abstractions.md#matrixoutput)
* **author:** Josh Klontz
* **properties:** None

---

#### mtxOutput



* **file:** output/mtx.cpp
* **inherits:** [Output](abstractions.md#output)
* **author:** Josh Klontz
* **properties:** None

---

#### nullOutput

Discards the scores.

* **file:** output/null.cpp
* **inherits:** [Output](abstractions.md#output)
* **author:** Josh Klontz
* **properties:** None

---

#### rankOutput

Outputs highest ranked matches with scores.

* **file:** output/rank.cpp
* **inherits:** [MatrixOutput](abstractions.md#matrixoutput)
* **author:** Scott Klum
* **properties:** None

---

#### rrOutput

Rank retrieval output.

* **file:** output/rr.cpp
* **inherits:** [MatrixOutput](abstractions.md#matrixoutput)
* **authors:** Josh Klontz, Scott Klum
* **properties:** None

---

#### tailOutput

The highest scoring matches.

* **file:** output/tail.cpp
* **inherits:** [Output](abstractions.md#output)
* **author:** Josh Klontz
* **properties:** None

---

#### txtOutput

Text file output.

* **file:** output/txt.cpp
* **inherits:** [MatrixOutput](abstractions.md#matrixoutput)
* **author:** Josh Klontz
* **properties:** None

## Video

---

#### AggregateFrames

Passes along n sequential frames to the next transform.

* **file:** video/aggregate.cpp
* **inherits:** [TimeVaryingTransform](abstractions.md#timevaryingtransform)
* **author:** Josh Klontz
* **properties:** None

---

#### DropFrames

Only use one frame every n frames.

* **file:** video/drop.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Austin Blanton
* **properties:** None

---

#### OpticalFlowTransform

Gets a one-channel dense optical flow from two images

* **file:** video/opticalflow.cpp
* **inherits:** [UntrainableMetaTransform](abstractions.md#untrainablemetatransform)
* **author:** Austin Blanton
* **properties:** None

