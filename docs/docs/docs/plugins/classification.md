---

# AdaBoostTransform

Wraps OpenCV's Ada Boost framework

* **file:** classification/adaboost.cpp
* **inherits:** [Transform](../cpp_api.md#transform)
* **author:** Scott Klum
* **properties:** None

---

# EBIFTransform

Face Recognition Using Early Biologically Inspired Features

* **file:** classification/ebif.cpp
* **inherits:** [UntrainableTransform](../cpp_api.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

# ForestTransform

Wraps OpenCV's random trees framework

* **file:** classification/forest.cpp
* **inherits:** [Transform](../cpp_api.md#transform)
* **author:** Scott Klum
* **properties:** None

---

# ForestInductionTransform

Wraps OpenCV's random trees framework to induce features

* **file:** classification/forest.cpp
* **inherits:** [ForestTransform](../cpp_api.md#foresttransform)
* **author:** Scott Klum
* **properties:** None

---

# IPC2013Initializer

Initializes Intel Perceptual Computing SDK 2013

* **file:** classification/ipc2013.cpp
* **inherits:** [Initializer](../cpp_api.md#initializer)
* **author:** Josh Klontz
* **properties:** None

---

# IPC2013FaceRecognitionTransfrom

Intel Perceptual Computing SDK 2013 Face Recognition

* **file:** classification/ipc2013.cpp
* **inherits:** [UntrainableTransform](../cpp_api.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

# EigenInitializer

Initialize Eigen

* **file:** classification/lda.cpp
* **inherits:** [Initializer](../cpp_api.md#initializer)
* **author:** Scott Klum
* **properties:** None

---

# PCATransform

Projects input into learned Principal Component Analysis subspace.

* **file:** classification/lda.cpp
* **inherits:** [Transform](../cpp_api.md#transform)
* **authors:** Brendan Klare, Josh Klontz
* **properties:** None

---

# RowWisePCATransform

PCA on each row.

* **file:** classification/lda.cpp
* **inherits:** [PCATransform](../cpp_api.md#pcatransform)
* **author:** Josh Klontz
* **properties:** None

---

# DFFSTransform

Computes Distance From Feature Space (DFFS)

* **file:** classification/lda.cpp
* **inherits:** [Transform](../cpp_api.md#transform)
* **author:** Josh Klontz
* **properties:** None

---

# LDATransform

Projects input into learned Linear Discriminant Analysis subspace.

* **file:** classification/lda.cpp
* **inherits:** [Transform](../cpp_api.md#transform)
* **authors:** Brendan Klare, Josh Klontz
* **properties:** None

---

# SparseLDATransform

Projects input into learned Linear Discriminant Analysis subspace

* **file:** classification/lda.cpp
* **inherits:** [Transform](../cpp_api.md#transform)
* **author:** Brendan Klare
* **properties:** None

---

# MLPTransform

Wraps OpenCV's multi-layer perceptron framework

* **file:** classification/mlp.cpp
* **inherits:** [MetaTransform](../cpp_api.md#metatransform)
* **author:** Scott Klum
* **properties:** None

---

# NT4Initializer

Initialize Neurotech SDK 4

* **file:** classification/nt4.cpp
* **inherits:** [Initializer](../cpp_api.md#initializer)
* **authors:** Josh Klontz, E. Taborsky
* **properties:** None

---

# NT4DetectFace

Neurotech face detection

* **file:** classification/nt4.cpp
* **inherits:** [UntrainableTransform](../cpp_api.md#untrainabletransform)
* **authors:** Josh Klontz, E. Taborsky
* **properties:** None

---

# NT4EnrollFace

Enroll face in Neurotech SDK 4

* **file:** classification/nt4.cpp
* **inherits:** [UntrainableTransform](../cpp_api.md#untrainabletransform)
* **author:** Josh Klontz
* **properties:** None

---

# NT4EnrollIris

Enroll iris in Neurotech SDK 4

* **file:** classification/nt4.cpp
* **inherits:** [UntrainableTransform](../cpp_api.md#untrainabletransform)
* **author:** E. Taborsky
* **properties:** None

---

# NT4Compare

Compare templates with Neurotech SDK 4

* **file:** classification/nt4.cpp
* **inherits:** [Distance](../cpp_api.md#distance)
* **authors:** Josh Klontz, E. Taborsky
* **properties:** None

---

# PP4Initializer

Initialize PittPatt 4

* **file:** classification/pp4.cpp
* **inherits:** [Initializer](../cpp_api.md#initializer)
* **author:** Josh Klontz
* **properties:** None

---

# PP4EnrollTransform

Enroll faces in PittPatt 4

* **file:** classification/pp4.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api.md#untrainablemetatransform)
* **author:** Josh Klontz
* **properties:** None

---

# PP5Initializer

Initialize PP5

* **file:** classification/pp5.cpp
* **inherits:** [Initializer](../cpp_api.md#initializer)
* **authors:** Josh Klontz, E. Taborsky
* **properties:** None

---

# PP5EnrollTransform

Enroll faces in PP5

* **file:** classification/pp5.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api.md#untrainablemetatransform)
* **authors:** Josh Klontz, E. Taborsky
* **properties:** None

---

# PP5CompareDistance

Compare templates with PP5

* **file:** classification/pp5.cpp
* **inherits:** [UntrainableDistance](../cpp_api.md#untrainabledistance)
* **authors:** Josh Klontz, E. Taborsky
* **properties:** None

---

# SVMTransform

C. Burges. "A tutorial on support vector machines for pattern recognition,"

* **file:** classification/svm.cpp
* **inherits:** [Transform](../cpp_api.md#transform)
* **author:** Josh Klontz
* **properties:** None

---

# SVMDistance

SVM Regression on template absolute differences.

* **file:** classification/svm.cpp
* **inherits:** [Distance](../cpp_api.md#distance)
* **author:** Josh Klontz
* **properties:** None

---

# TurkClassifierTransform

Convenience class for training turk attribute regressors

* **file:** classification/turk.cpp
* **inherits:** [Transform](../cpp_api.md#transform)
* **author:** Josh Klontz
* **properties:** None

