# AbsDiffTransform

Take the absolute difference of two matrices.
 

* **file:** imgproc/absdiff.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# AbsTransform

Computes the absolute value of each element.
 

* **file:** imgproc/abs.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# AdaptiveThresholdTransform

Wraps OpenCV's adaptive thresholding.
 

* **file:** imgproc/adaptivethreshold.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Scott Klum][sklum]
* **see:** [http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html](http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html)
* **properties:** None


---

# AffineTransform

Performs a two or three point registration.
 

* **file:** imgproc/affine.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# AndTransform

Logical AND of two matrices.
 

* **file:** imgproc/and.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# ApplyMaskTransform

Applies a mask from the metadata.
 

* **file:** imgproc/applymask.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Austin Blanton][imaus10]
* **properties:** None


---

# BayesianQuantizationTransform

Quantize into a space where L1 distance approximates log-likelihood.
 

* **file:** imgproc/bayesianquantization.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# BinarizeTransform

Approximate floats as signed bit.
 

* **file:** imgproc/binarize.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# BlendTransform

Alpha-blend two matrices
 

* **file:** imgproc/blend.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# BlurTransform

Gaussian blur
 

* **file:** imgproc/blur.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# BuildScalesTransform

DOCUMENT ME
 

* **file:** imgproc/multiscale.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author(s):** [Austin Blanton][imaus10]
* **properties:** None


---

# ByRowTransform

Turns each row into its own matrix.
 

* **file:** imgproc/byrow.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# CannyTransform

Wrapper to OpenCV Canny edge detector
 

* **file:** imgproc/canny.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Scott Klum][sklum]
* **see:** [http://docs.opencv.org/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html](http://docs.opencv.org/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html)
* **properties:** None


---

# CatColsTransform

Concatenates all input matrices by column into a single matrix.
Use after a fork to concatenate two feature matrices by column.
 

* **file:** imgproc/catcols.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Austin Blanton][imaus10]
* **properties:** None


---

# CatRowsTransform

Concatenates all input matrices by row into a single matrix.
All matricies must have the same column counts.
 

* **file:** imgproc/catrows.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# CatTransform

Concatenates all input matrices into a single matrix.
 

* **file:** imgproc/cat.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# CenterTransform

Normalize each dimension based on training data.
 

* **file:** imgproc/center.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# ContrastEqTransform

Perform contrast equalization
 

* **file:** imgproc/contrasteq.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **read:**

	1. *Xiaoyang Tan; Triggs, B.;*
	 **"Enhanced Local Texture Feature Sets for Face Recognition Under Difficult Lighting Conditions,"**
	 Image Processing, IEEE Transactions on , vol.19, no.6, pp.1635-1650, June 2010

* **properties:** None


---

# CropBlackTransform

Crop out black borders
 

* **file:** imgproc/cropblack.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# CropFromMaskTransform

Crops image based on mask metadata
 

* **file:** imgproc/cropfrommask.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Brendan Klare][bklare]
* **properties:** None


---

# CropSquareTransform

Trim the image so the width and the height are the same size.
 

* **file:** imgproc/cropsquare.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# CropTransform

Crops about the specified region of interest.
 

* **file:** imgproc/crop.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# CryptographicHashTransform

Wraps QCryptographicHash
 

* **file:** imgproc/cryptographichash.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **see:** [http://doc.qt.io/qt-5/qcryptographichash.html](http://doc.qt.io/qt-5/qcryptographichash.html)
* **properties:** None


---

# CvtFloatTransform

Convert to floating point format.
 

* **file:** imgproc/cvtfloat.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# CvtTransform

Colorspace conversion.
 

* **file:** imgproc/cvt.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# CvtUCharTransform

Convert to uchar format
 

* **file:** imgproc/cvtuchar.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# DiscardAlphaTransform

Drop the alpha channel (if exists).
 

* **file:** imgproc/discardalpha.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Austin Blanton][imaus10]
* **properties:** None


---

# DivTransform

Enforce a multiple of n columns.
 

* **file:** imgproc/div.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# DoGTransform

Difference of gaussians
 

* **file:** imgproc/dog.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# DownsampleTransform

Downsample the rows and columns of a matrix.
 

* **file:** imgproc/downsample.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Lacey Best-Rowden][lbestrowden]
* **properties:** None


---

# DupTransform

Duplicates the [Template](../cpp_api/template/template.md) data.
 

* **file:** imgproc/dup.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# EnsureChannelsTransform

Enforce the matrix has a certain number of channels by adding or removing channels.
 

* **file:** imgproc/ensurechannels.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# EqualizeHistTransform

Histogram equalization
 

* **file:** imgproc/equalizehist.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# FlipTransform

Flips the image about an axis.
 

* **file:** imgproc/flip.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# FloodTransform

Fill black pixels with the specified color.
 

* **file:** imgproc/flood.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# GaborJetTransform

A vector of gabor wavelets applied at a point.
 

* **file:** imgproc/gabor.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# GaborTransform

Implements a Gabor Filter
 

* **file:** imgproc/gabor.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **see:** [http://en.wikipedia.org/wiki/Gabor_filter](http://en.wikipedia.org/wiki/Gabor_filter)
* **properties:** None


---

# GammaTransform

Gamma correction
 

* **file:** imgproc/gamma.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# GradientMaskTransform

Masks image according to pixel change.
 

* **file:** imgproc/gradientmask.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# GradientTransform

Computes magnitude and/or angle of image.
 

* **file:** imgproc/gradient.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# GroupTransform

Group all input matrices into a single matrix.

Similar to CatTransfrom but groups every _size_ adjacent matricies.
 

* **file:** imgproc/group.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# HeatmapTransform

DOCUMENT ME
 

* **file:** imgproc/heatmap.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Unknown][unknown]
* **properties:** None


---

# HistBinTransform

Quantizes the values into bins.
 

* **file:** imgproc/histbin.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# HistEqQuantizationTransform

Approximate floats as uchar with different scalings for each dimension.
 

* **file:** imgproc/histeqquantization.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# HistTransform

Histograms the matrix
 

* **file:** imgproc/hist.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# HoGDescriptorTransform

OpenCV HOGDescriptor wrapper
 

* **file:** imgproc/hog.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Austin Blanton][imaus10]
* **see:** [http://docs.opencv.org/modules/gpu/doc/object_detection.html](http://docs.opencv.org/modules/gpu/doc/object_detection.html)
* **properties:** None


---

# InpaintTransform

Wraps OpenCV inpainting
 

* **file:** imgproc/inpaint.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **see:** [http://docs.opencv.org/modules/photo/doc/inpainting.html](http://docs.opencv.org/modules/photo/doc/inpainting.html)
* **properties:** None


---

# IntegralHistTransform

An integral histogram
 

* **file:** imgproc/integralhist.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# IntegralSamplerTransform

Sliding window feature extraction from a multi-channel integral image.
 

* **file:** imgproc/integralsampler.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# IntegralSlidingWindowTransform

Overloads SlidingWindowTransform for integral images that should be
sampled at multiple scales.
 

* **file:** imgproc/slidingwindow.cpp
* **inherits:** [SlidingWindowTransform](#slidingwindowtransform)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# IntegralTransform

Computes integral image.
 

* **file:** imgproc/integral.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# KernelHashTransform

Kernel hash
 

* **file:** imgproc/kernelhash.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# KeyPointDescriptorTransform

Wraps OpenCV Key Point Descriptor
 

* **file:** imgproc/keypointdescriptor.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **see:** [http://docs.opencv.org/modules/features2d/doc/common_interfaces_of_feature_detectors.html](http://docs.opencv.org/modules/features2d/doc/common_interfaces_of_feature_detectors.html)
* **properties:** None


---

# LBPTransform

Convert the image into a feature vector using Local Binary Patterns
 

* **file:** imgproc/lbp.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **read:**

	1. *Ahonen, T.; Hadid, A.; Pietikainen, M.;*
	 **"Face Description with Local Binary Patterns: Application to Face Recognition"**
	 Pattern Analysis and Machine Intelligence, IEEE Transactions, vol.28, no.12, pp.2037-2041, Dec. 2006

* **properties:** None


---

# LTPTransform

DOCUMENT ME
 

* **file:** imgproc/ltp.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Brendan Klare][bklare], [Josh Klontz][jklontz]
* **read:**

	1. *Tan, Xiaoyang, and Bill Triggs.*
	 **"Enhanced local texture feature sets for face recognition under difficult lighting conditions."**
	 Analysis and Modeling of Faces and Gestures. Springer Berlin Heidelberg, 2007. 168-182.

* **properties:** None


---

# LargestConvexAreaTransform

Set the template's label to the area of the largest convex hull.
 

* **file:** imgproc/largestconvexarea.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# LimitSizeTransform

Limit the size of the template
 

* **file:** imgproc/limitsize.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# MAddTransform

dst = a src+b
 

* **file:** imgproc/madd.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# MaskTransform

Applies an eliptical mask
 

* **file:** imgproc/mask.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# MatStatsTransform

Statistics
 

* **file:** imgproc/matstats.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# MeanFillTransform

Fill 0 pixels with the mean of non-0 pixels.
 

* **file:** imgproc/meanfill.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# MeanTransform

Computes the mean of a set of templates.

Suitable for visualization only as it sets every projected template to the mean template.
 

* **file:** imgproc/mean.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author(s):** [Scott Klum][sklum]
* **properties:** None


---

# MergeTransform

Wraps OpenCV merge
 

* **file:** imgproc/merge.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **see:** [http://docs.opencv.org/modules/core/doc/operations_on_arrays.html#merge](http://docs.opencv.org/modules/core/doc/operations_on_arrays.html#merge)
* **properties:** None


---

# MorphTransform

Morphological operator
 

* **file:** imgproc/morph.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# NLMeansDenoisingTransform

Wraps OpenCV Non-Local Means Denoising
 

* **file:** imgproc/denoising.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **see:** [http://docs.opencv.org/modules/photo/doc/denoising.html](http://docs.opencv.org/modules/photo/doc/denoising.html)
* **properties:** None


---

# NormalizeTransform

Normalize matrix to unit length
 

* **file:** imgproc/normalize.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:**

	Property | Type | Description
	--- | --- | ---
	NormType | enum | Values are:<ul><li>NORM_INF</li><li>NORM_L1</li><li>NORM_L2</li><li>NORM_MINMAX</li></ul>
	ByRow | bool | If true normalize each row independently otherwise normalize the entire matrix.
	alpha | int | Lower bound if using NORM_MINMAX. Value to normalize to otherwise.
	beta | int | Upper bound if using NORM_MINMAX. Not used otherwise.
	squareRoot | bool | If true compute the signed square root of the output after normalization.

---

# OrigLinearRegressionTransform

Prediction with magic numbers from jmp; must get input as blue;green;red
 

* **file:** imgproc/origlinearregression.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [E. Taborsky][mmtaborsky]
* **properties:** None


---

# PackTransform

Compress two uchar into one uchar.
 

* **file:** imgproc/pack.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# PowTransform

Raise each element to the specified power.
 

* **file:** imgproc/pow.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# ProductQuantizationDistance

Distance in a product quantized space
 

* **file:** imgproc/productquantization.cpp
* **inherits:** [UntrainableDistance](../cpp_api/untrainabledistance/untrainabledistance.md)
* **author(s):** [Josh Klontz][jklontz]
* **read:**

	1. *Jegou, Herve, Matthijs Douze, and Cordelia Schmid.*
	 **"Product quantization for nearest neighbor search."**
	 Pattern Analysis and Machine Intelligence, IEEE Transactions on 33.1 (2011): 117-128

* **properties:** None


---

# ProductQuantizationTransform

Product quantization
 

* **file:** imgproc/productquantization.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author(s):** [Josh Klontz][jklontz]
* **read:**

	1. *Jegou, Herve, Matthijs Douze, and Cordelia Schmid.*
	 **"Product quantization for nearest neighbor search."**
	 Pattern Analysis and Machine Intelligence, IEEE Transactions on 33.1 (2011): 117-128

* **properties:** None


---

# QuantizeTransform

Approximate floats as uchar.
 

* **file:** imgproc/quantize.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# RGTransform

Normalized RG color space.
 

* **file:** imgproc/rg.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# ROIFromPtsTransform

Crops the rectangular regions of interest from given points and sizes.
 

* **file:** imgproc/roifrompoints.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Austin Blanton][imaus10]
* **properties:** None


---

# ROITransform

Crops the rectangular regions of interest.
 

* **file:** imgproc/roi.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# RankTransform

Converts each element to its rank-ordered value.
 

* **file:** imgproc/rank.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# RectRegionsTransform

Subdivide matrix into rectangular subregions.
 

* **file:** imgproc/rectregions.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# RecursiveIntegralSamplerTransform

Construct [Template](../cpp_api/template/template.md) in a recursive decent manner.
 

* **file:** imgproc/recursiveintegralsampler.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# RedLinearRegressionTransform

Prediction using only the red wavelength; magic numbers from jmp
 

* **file:** imgproc/redlinearregression.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [E. Taborsky][mmtaborsky]
* **properties:** None


---

# ReshapeTransform

Reshape each matrix to the specified number of rows.
 

* **file:** imgproc/reshape.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# ResizeTransform

Resize the template
 

* **file:** imgproc/resize.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:**

	Property | Type | Description
	--- | --- | ---
	method | enum | Resize method. Good options are:<ul><li>Area should be used for shrinking an image</li><li>Cubic for slow but accurate enlargment</li><li>Bilin for fast enlargement</li></ul>
	preserveAspect | bool | If true, the image will be sized per specification, but a border will be applied to preserve aspect ratio.

---

# RevertAffineTransform

Designed for use after eye detection + Stasm, this will
revert the detected landmarks to the original coordinate space
before affine alignment to the stasm mean shape. The storeAffine
parameter must be set to true when calling AffineTransform before this.
 

* **file:** imgproc/revertaffine.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Brendan Klare][bklare]
* **properties:** None


---

# RndPointTransform

Generates a random landmark.
 

* **file:** imgproc/rndpoint.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# RndRegionTransform

Selects a random region.
 

* **file:** imgproc/rndregion.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# RndRotateTransform

Randomly rotates an image in a specified range.
 

* **file:** imgproc/rndrotate.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Scott Klum][sklum]
* **properties:** None


---

# RndSubspaceTransform

Generates a random subspace.
 

* **file:** imgproc/rndsubspace.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# RootNormTransform

dst=sqrt(norm_L1(src)) proposed as RootSIFT (see paper)
 

* **file:** imgproc/rootnorm.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **read:**

	1. *Arandjelovic, Relja, and Andrew Zisserman.*
	 **"Three things everyone should know to improve object retrieval."**
	 Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on. IEEE, 2012.

* **properties:** None


---

# RowWiseMeanCenterTransform

Remove the row-wise training set average.
 

* **file:** imgproc/rowwisemeancenter.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# SIFTDescriptorTransform

Specialize wrapper OpenCV SIFT wrapper
 

* **file:** imgproc/sift.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **see:** [http://docs.opencv.org/modules/nonfree/doc/feature_detection.html](http://docs.opencv.org/modules/nonfree/doc/feature_detection.html)
* **properties:** None


---

# SampleFromMaskTransform

Samples pixels from a mask.
 

* **file:** imgproc/samplefrommask.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Scott Klum][sklum]
* **properties:** None


---

# ScaleTransform

Scales using the given factor
 

* **file:** imgproc/scale.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Scott Klum][sklum]
* **properties:** None


---

# SkinMaskTransform

Make a mask over skin in an image
 

* **file:** imgproc/skinmask.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **see:** [http://worldofcameras.wordpress.com/tag/skin-detection-opencv/](http://worldofcameras.wordpress.com/tag/skin-detection-opencv/)
* **properties:** None


---

# SlidingWindowTransform

Applies a transform to a sliding window.
Discards negative detections.
 

* **file:** imgproc/slidingwindow.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author(s):** [Austin Blanton][imaus10]
* **properties:** None


---

# SplitChannelsTransform

Split a multi-channel matrix into several single-channel matrices.
 

* **file:** imgproc/splitchannels.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# SubdivideTransform

Divide the matrix into 4 smaller matricies of equal size.
 

* **file:** imgproc/subdivide.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# SubtractTransform

Subtract two matrices.
 

* **file:** imgproc/subtract.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# ThresholdTransform

Wraps OpenCV's adaptive thresholding.
 

* **file:** imgproc/threshold.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Scott Klum][sklum]
* **see:** [http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html](http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html)
* **properties:** None


---

# TransposeTransform

Get the transpose of the [Template](../cpp_api/template/template.md) matrix
 

* **file:** imgproc/transpose.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Unknown][unknown]
* **properties:** None


---

# WatershedSegmentationTransform

Applies watershed segmentation.
 

* **file:** imgproc/watershedsegmentation.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author(s):** [Austin Blanton][imaus10]
* **properties:** None


---

