---

# AttributeDistance

Attenuation function based distance from attributes

* **file:** distance/attribute.cpp
* **inherits:** [UntrainableDistance](../cpp_api.md#untrainabledistance)
* **author:** Scott Klum
* **properties:** None

---

# BayesianQuantizationDistance

Bayesian quantization distance

* **file:** distance/bayesianquantization.cpp
* **inherits:** [Distance](../cpp_api.md#distance)
* **author:** Josh Klontz
* **properties:** None

---

# ByteL1Distance

Fast 8-bit L1 distance

* **file:** distance/byteL1.cpp
* **inherits:** [UntrainableDistance](../cpp_api.md#untrainabledistance)
* **author:** Josh Klontz
* **properties:** None

---

# CrossValidateDistance

Cross validate a distance metric.

* **file:** distance/crossvalidate.cpp
* **inherits:** [UntrainableDistance](../cpp_api.md#untrainabledistance)
* **author:** Josh Klontz
* **properties:** None

---

# DefaultDistance

DistDistance wrapper.

* **file:** distance/default.cpp
* **inherits:** [UntrainableDistance](../cpp_api.md#untrainabledistance)
* **author:** Josh Klontz
* **properties:** None

---

# DistDistance

Standard distance metrics

* **file:** distance/dist.cpp
* **inherits:** [UntrainableDistance](../cpp_api.md#untrainabledistance)
* **author:** Josh Klontz
* **properties:** None

---

# FilterDistance

Checks target metadata against filters.

* **file:** distance/filter.cpp
* **inherits:** [UntrainableDistance](../cpp_api.md#untrainabledistance)
* **author:** Josh Klontz
* **properties:** None

---

# FuseDistance

Fuses similarity scores across multiple matrices of compared templates

* **file:** distance/fuse.cpp
* **inherits:** [Distance](../cpp_api.md#distance)
* **author:** Scott Klum
* **properties:** None

---

# HalfByteL1Distance

Fast 4-bit L1 distance

* **file:** distance/halfbyteL1.cpp
* **inherits:** [UntrainableDistance](../cpp_api.md#untrainabledistance)
* **author:** Josh Klontz
* **properties:** None

---

# HeatMapDistance

1v1 heat map comparison

* **file:** distance/heatmap.cpp
* **inherits:** [Distance](../cpp_api.md#distance)
* **author:** Scott Klum
* **properties:** None

---

# IdenticalDistance

Returns

* **file:** distance/identical.cpp
* **inherits:** [UntrainableDistance](../cpp_api.md#untrainabledistance)
* **author:** Josh Klontz
* **properties:** None

---

# KeyPointMatcherDistance

Wraps OpenCV Key Point Matcher

* **file:** distance/keypointmatcher.cpp
* **inherits:** [UntrainableDistance](../cpp_api.md#untrainabledistance)
* **author:** Josh Klontz
* **properties:** None

---

# L1Distance

L1 distance computed using eigen.

* **file:** distance/L1.cpp
* **inherits:** [UntrainableDistance](../cpp_api.md#untrainabledistance)
* **author:** Josh Klontz
* **properties:** None

---

# L2Distance

L2 distance computed using eigen.

* **file:** distance/L2.cpp
* **inherits:** [UntrainableDistance](../cpp_api.md#untrainabledistance)
* **author:** Josh Klontz
* **properties:** None

---

# MatchProbabilityDistance

Match Probability

* **file:** distance/matchprobability.cpp
* **inherits:** [Distance](../cpp_api.md#distance)
* **author:** Josh Klontz
* **properties:** None

---

# MetadataDistance

Checks target metadata against query metadata.

* **file:** distance/metadata.cpp
* **inherits:** [UntrainableDistance](../cpp_api.md#untrainabledistance)
* **author:** Scott Klum
* **properties:** None

---

# NegativeLogPlusOneDistance

Returns -log(distance(a,b)+1)

* **file:** distance/neglogplusone.cpp
* **inherits:** [UntrainableDistance](../cpp_api.md#untrainabledistance)
* **author:** Josh Klontz
* **properties:** None

---

# OnlineDistance

Online distance metric to attenuate match scores across multiple frames

* **file:** distance/online.cpp
* **inherits:** [UntrainableDistance](../cpp_api.md#untrainabledistance)
* **author:** Brendan klare
* **properties:** None

---

# PipeDistance

Distances in series.

* **file:** distance/pipe.cpp
* **inherits:** [Distance](../cpp_api.md#distance)
* **author:** Josh Klontz
* **properties:** None

---

# RejectDistance

Sets distance to -FLOAT_MAX if a target template has/doesn't have a key.

* **file:** distance/reject.cpp
* **inherits:** [UntrainableDistance](../cpp_api.md#untrainabledistance)
* **author:** Scott Klum
* **properties:** None

---

# SumDistance

Sum match scores across multiple distances

* **file:** distance/sum.cpp
* **inherits:** [UntrainableDistance](../cpp_api.md#untrainabledistance)
* **author:** Scott Klum
* **properties:** None

---

# TurkDistance

Unmaps Turk HITs to be compared against query mats

* **file:** distance/turk.cpp
* **inherits:** [UntrainableDistance](../cpp_api.md#untrainabledistance)
* **author:** Scott Klum
* **properties:** None

---

# UnitDistance

Linear normalizes of a distance so the mean impostor score is 0 and the mean genuine score is 1.

* **file:** distance/unit.cpp
* **inherits:** [Distance](../cpp_api.md#distance)
* **author:** Josh Klontz
* **properties:** None

