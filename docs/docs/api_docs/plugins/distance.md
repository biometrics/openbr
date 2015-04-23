# AttributeDistance

Attenuation function based distance from attributes
 

* **file:** distance/attribute.cpp
* **inherits:** [UntrainableDistance](../cpp_api/untrainabledistance/untrainabledistance.md)
* **author(s):** [Scott Klum][sklum]
* **properties:** None


---

# BayesianQuantizationDistance

Bayesian quantization [Distance](../cpp_api/distance/distance.md)
 

* **file:** distance/bayesianquantization.cpp
* **inherits:** [Distance](../cpp_api/distance/distance.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# ByteL1Distance

Fast 8-bit L1 distance
 

* **file:** distance/byteL1.cpp
* **inherits:** [UntrainableDistance](../cpp_api/untrainabledistance/untrainabledistance.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# CrossValidateDistance

Cross validate a [Distance](../cpp_api/distance/distance.md) metric.
 

* **file:** distance/crossvalidate.cpp
* **inherits:** [UntrainableDistance](../cpp_api/untrainabledistance/untrainabledistance.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# DefaultDistance

DistDistance wrapper.
 

* **file:** distance/default.cpp
* **inherits:** [UntrainableDistance](../cpp_api/untrainabledistance/untrainabledistance.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# DistDistance

Standard [Distance](../cpp_api/distance/distance.md) metrics
 

* **file:** distance/dist.cpp
* **inherits:** [UntrainableDistance](../cpp_api/untrainabledistance/untrainabledistance.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# FilterDistance

Checks target metadata against filters.
 

* **file:** distance/filter.cpp
* **inherits:** [UntrainableDistance](../cpp_api/untrainabledistance/untrainabledistance.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# FuseDistance

Fuses similarity scores across multiple matrices of compared [Template](../cpp_api/template/template.md)
 

* **file:** distance/fuse.cpp
* **inherits:** [Distance](../cpp_api/distance/distance.md)
* **author(s):** [Scott Klum][sklum]
* **properties:**

	Property | Type | Description
	--- | --- | ---
	Operation | enum | Possible values are:<ul><li>Mean</li><li>sum</li><li>min</li><li>max</li></ul>.

---

# HalfByteL1Distance

Fast 4-bit L1 distance
 

* **file:** distance/halfbyteL1.cpp
* **inherits:** [UntrainableDistance](../cpp_api/untrainabledistance/untrainabledistance.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# HeatMapDistance

1v1 heat map comparison
 

* **file:** distance/heatmap.cpp
* **inherits:** [Distance](../cpp_api/distance/distance.md)
* **author(s):** [Scott Klum][sklum]
* **properties:** None


---

# IdenticalDistance

Returns true if the [Template](../cpp_api/template/template.md) are identical, false otherwise.
 

* **file:** distance/identical.cpp
* **inherits:** [UntrainableDistance](../cpp_api/untrainabledistance/untrainabledistance.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# KeyPointMatcherDistance

Wraps OpenCV Key Point Matcher
 

* **file:** distance/keypointmatcher.cpp
* **inherits:** [UntrainableDistance](../cpp_api/untrainabledistance/untrainabledistance.md)
* **author(s):** [Josh Klontz][jklontz]
* **see:** [http://docs.opencv.org/modules/features2d/doc/common_interfaces_of_feature_detectors.html](http://docs.opencv.org/modules/features2d/doc/common_interfaces_of_feature_detectors.html)
* **properties:** None


---

# L1Distance

L1 distance computed using eigen.
 

* **file:** distance/L1.cpp
* **inherits:** [UntrainableDistance](../cpp_api/untrainabledistance/untrainabledistance.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# L2Distance

L2 distance computed using eigen.
 

* **file:** distance/L2.cpp
* **inherits:** [UntrainableDistance](../cpp_api/untrainabledistance/untrainabledistance.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# MatchProbabilityDistance

Match Probability
 

* **file:** distance/matchprobability.cpp
* **inherits:** [Distance](../cpp_api/distance/distance.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# MetadataDistance

Checks target metadata against query metadata.
 

* **file:** distance/metadata.cpp
* **inherits:** [UntrainableDistance](../cpp_api/untrainabledistance/untrainabledistance.md)
* **author(s):** [Scott Klum][sklum]
* **properties:** None


---

# NegativeLogPlusOneDistance

Returns -log(distance(a,b)+1)
 

* **file:** distance/neglogplusone.cpp
* **inherits:** [UntrainableDistance](../cpp_api/untrainabledistance/untrainabledistance.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# OnlineDistance

Online [Distance](../cpp_api/distance/distance.md) metric to attenuate match scores across multiple frames
 

* **file:** distance/online.cpp
* **inherits:** [UntrainableDistance](../cpp_api/untrainabledistance/untrainabledistance.md)
* **author(s):** [Brendan klare][bklare]
* **properties:** None


---

# PipeDistance

Distances in series.

The [Template](../cpp_api/template/template.md) are compared using each [Distance](../cpp_api/distance/distance.md) in order.
If the result of the comparison with any given distance is -FLOAT_MAX then this result is returned early.
Otherwise the returned result is the value of comparing the [Template](../cpp_api/template/template.md) using the last [Distance](../cpp_api/distance/distance.md).
 

* **file:** distance/pipe.cpp
* **inherits:** [Distance](../cpp_api/distance/distance.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# RejectDistance

Sets [Distance](../cpp_api/distance/distance.md) to -FLOAT_MAX if a target [Template](../cpp_api/template/template.md) has/doesn't have a key.
 

* **file:** distance/reject.cpp
* **inherits:** [UntrainableDistance](../cpp_api/untrainabledistance/untrainabledistance.md)
* **author(s):** [Scott Klum][sklum]
* **properties:** None


---

# SVMDistance

SVM Regression on [Template](../cpp_api/template/template.md) absolute differences.
 

* **file:** distance/svm.cpp
* **inherits:** [Distance](../cpp_api/distance/distance.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# SumDistance

Sum match scores across multiple [Distance](../cpp_api/distance/distance.md)
 

* **file:** distance/sum.cpp
* **inherits:** [UntrainableDistance](../cpp_api/untrainabledistance/untrainabledistance.md)
* **author(s):** [Scott Klum][sklum]
* **properties:** None


---

# TurkDistance

Unmaps Turk HITs to be compared against query mats
 

* **file:** distance/turk.cpp
* **inherits:** [UntrainableDistance](../cpp_api/untrainabledistance/untrainabledistance.md)
* **author(s):** [Scott Klum][sklum]
* **properties:** None


---

# UnitDistance

Linear normalizes of a [Distance](../cpp_api/distance/distance.md) so the mean impostor score is 0 and the mean genuine score is 1.
 

* **file:** distance/unit.cpp
* **inherits:** [Distance](../cpp_api/distance/distance.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# ZScoreDistance

DOCUMENT ME
 

* **file:** distance/zscore.cpp
* **inherits:** [Distance](../cpp_api/distance/distance.md)
* **author(s):** [Unknown][unknown]
* **properties:** None


---

