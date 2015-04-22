# AdaBoostTransform

Wraps OpenCV's Ada Boost framework

* **file:** classification/adaboost.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **see:** [http://docs.opencv.org/modules/ml/doc/boosting.html](http://docs.opencv.org/modules/ml/doc/boosting.html)
* **author:** Scott Klum
* **properties:**

Property | Type | Description
--- | --- | ---
type | enum | Type of Adaboost to perform. Options are:<ul><li>Discrete</li><li>Real</li><li>Logit</li><li>Gentle</li></ul>Default is Real.
splitCriteria | enum | Splitting criteria used to choose optimal splits during a weak tree construction. Options are:<ul><li>Default</li><li>Gini</li><li>Misclass</li><li>Sqerr</li></ul>Default is Default.
weakCount | int | Maximum number of weak classifiers per stage. Default is 100.
trimRate | float | A threshold between 0 and 1 used to save computational time. Samples with summary weight
folds | int | OpenCV parameter variable. Default value is 0.
maxDepth | int | Maximum height of each weak classifier tree. Default is 1 (stumps).
returnConfidence | bool | Return the confidence value of the classification or the class value of the classification. Default is true (return confidence value).
overwriteMat | bool | If true, the output template will be a 1x1 matrix with value equal to the confidence or classification (depending on returnConfidence). If false the output template will be the same as the input template. Default is true.
inputVariable | QString | Metadata variable storing the label for each template. Default is "Label".
outputVariable | QString | Metadata variable to store the confidence or classification of each template (depending on returnConfidence). If overwriteMat is true nothing will be written here. Default is "".

---

# DFFSTransform

Computes Distance From Feature Space (DFFS)

* **file:** classification/lda.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author:** Josh Klontz
* **properties:**

Property | Type | Description
--- | --- | ---
keep | float | Sets PCA keep property. Default is 0.95.

---

# EBIFTransform

Face Recognition Using Early Biologically Inspired Features

* **file:** classification/ebif.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **see:** [Min Li (IBM China Research Lab, China), Nalini Ratha (IBM Watson Research Center, USA), Weihong Qian (IBM China Research Lab, China), Shenghua Bao (IBM China Research Lab, China), Zhong Su (IBM China Research Lab, China)](#min li (ibm china research lab, china), nalini ratha (ibm watson research center, usa), weihong qian (ibm china research lab, china), shenghua bao (ibm china research lab, china), zhong su (ibm china research lab, china))
* **author:** Josh Klontz
* **properties:**

Property | Type | Description
--- | --- | ---
N | int | The number of scales. Default is 6.
M | int | The number of orientations between 0 and pi. Default is 9.

---

# ForestInductionTransform

Wraps OpenCV's random trees framework to induce features

* **file:** classification/forest.cpp
* **inherits:** [ForestTransform](#foresttransform)
* **see:** [https://lirias.kuleuven.be/bitstream/123456789/316661/1/icdm11-camready.pdf](https://lirias.kuleuven.be/bitstream/123456789/316661/1/icdm11-camready.pdf)
* **author:** Scott Klum
* **properties:**

Property | Type | Description
--- | --- | ---
useRegressionValue | bool | SCOTT FILL ME IN.

---

# ForestTransform

Wraps OpenCV's random trees framework

* **file:** classification/forest.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **see:** [http://docs.opencv.org/modules/ml/doc/random_trees.html](http://docs.opencv.org/modules/ml/doc/random_trees.html)
* **author:** Scott Klum
* **properties:**

Property | Type | Description
--- | --- | ---
classification | bool | If true the labels are expected to be categorical. Otherwise they are expected to be numerical. Default is true.
splitPercentage | float | Used to calculate the minimum number of samples per split in a random tree. The minimum number of samples is calculated as the number of samples x splitPercentage. Default is 0.01.
maxDepth | int | The maximum depth of each decision tree. Default is std::numeric_limits<int>::max() and typically should be set by the user.
maxTrees | int | The maximum number of trees in the forest. Default is 10.
forestAccuracy | float | A sufficient accuracy for the forest for training to terminate. Used if termCrit is EPS or Both. Default is 0.1.
returnConfidence | bool | If both classification and returnConfidence are use a fuzzy class label as the output of the forest. Default is true.
overwriteMat | bool | If true set dst to be a 1x1 Mat with the forest response as its value. Otherwise append the forest response to metadata using outputVariable as a key. Default is true.
inputVariable | QString | The metadata key for each templates label. Default is "Label".
outputVariable | QString | The metadata key for the forest response if overwriteMat is false. Default is "".
weight | bool | If true and classification is true the random forest will use prior accuracies. Default is false.
termCrit | enum | Termination criteria for training the random forest. Options are Iter, EPS and Both. Iter terminates when the maximum number of trees is reached. EPS terminates when forestAccuracy is met. Both terminates when either is true. Default is Iter.

---

# IPC2013FaceRecognitionTransform

Intel Perceptual Computing SDK 2013 Face Recognition

* **file:** classification/ipc2013.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author:** Josh Klontz
* **properties:** None


---

# LDATransform

Projects input into learned Linear Discriminant Analysis subspace.

* **file:** classification/lda.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **authors:** Brendan Klare, Josh Klontz
* **properties:**

Property | Type | Description
--- | --- | ---
pcaKeep | float | BRENDAN OR JOSH FILL ME IN. Default is 0.98.
pcaWhiten | bool | BRENDAN OR JOSH FILL ME IN. Default is false.
directLDA | int | BRENDAN OR JOSH FILL ME IN. Default is 0.
directDrop | float | BRENDAN OR JOSH FILL ME IN. Default is 0.1.
inputVariable | QString | BRENDAN OR JOSH FILL ME IN. Default is "Label".
isBinary | bool | BRENDAN OR JOSH FILL ME IN. Default is false.
normalize | bool | BRENDAN OR JOSH FILL ME IN. Default is true.

---

# MLPTransform

Wraps OpenCV's multi-layer perceptron framework

* **file:** classification/mlp.cpp
* **inherits:** [MetaTransform](../cpp_api/metatransform/metatransform.md)
* **see:** [http://docs.opencv.org/modules/ml/doc/neural_networks.html](http://docs.opencv.org/modules/ml/doc/neural_networks.html)
* **author:** Scott Klum
* **properties:**

Property | Type | Description
--- | --- | ---
kernel | enum | Type of MLP kernel to use. Options are Identity, Sigmoid, Gaussian. Default is Sigmoid.
alpha | float | Determines activation function for neural network. See OpenCV documentation for more details. Default is 1.
beta | float | Determines activation function for neural network. See OpenCV documentation for more details. Default is 1.
inputVariables | QStringList | Metadata keys for the labels associated with each template. There should be the same number of keys in the list as there are neurons in the final layer. Default is QStringList().
outputVariables | QStringList | Metadata keys to store the output of the neural network. There should be the same number of keys in the list as there are neurons in the final layer. Default is QStringList().
neuronsPerLayer | QList<int> | The number of neurons in each layer of the net. Default is QList<int>() << 1 << 1.

---

# NT4Compare

Compare templates with Neurotech SDK 4

* **file:** classification/nt4.cpp
* **inherits:** [Distance](../cpp_api/distance/distance.md)
* **authors:** Josh Klontz, E. Taborsky
* **properties:** None


---

# NT4DetectFace

Neurotech face detection

* **file:** classification/nt4.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **authors:** Josh Klontz, E. Taborsky
* **properties:** None


---

# NT4EnrollFace

Enroll face in Neurotech SDK 4

* **file:** classification/nt4.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author:** Josh Klontz
* **properties:** None


---

# NT4EnrollIris

Enroll iris in Neurotech SDK 4

* **file:** classification/nt4.cpp
* **inherits:** [UntrainableTransform](../cpp_api/untrainabletransform/untrainabletransform.md)
* **author:** E. Taborsky
* **properties:** None


---

# PCATransform

Projects input into learned Principal Component Analysis subspace.

* **file:** classification/lda.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **authors:** Brendan Klare, Josh Klontz
* **properties:**

Property | Type | Description
--- | --- | ---
keep | float | Options are:<ul><li>keep < 0 - All eigenvalues are retained</li><li>keep == 0 - No PCA is performed and the eigenvectors form an identity matrix</li><li>0 < keep < 1 - Keep is the fraction of the variance to retain</li><li>keep >= 1 - keep is the number of leading eigenvectors to retain</li></ul>Default is 0.95.
drop | int | BRENDAN OR JOSH FILL ME IN. Default is 0.
whiten | bool | BRENDAN OR JOSH FILL ME IN. Default is false.

---

# PP4Compare

Compare faces using PittPatt 4.

* **file:** classification/pp4.cpp
* **inherits:** [Distance](../cpp_api/distance/distance.md)
* **author:** Josh Klontz
* **properties:** None


---

# PP4EnrollTransform

Enroll faces in PittPatt 4

* **file:** classification/pp4.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author:** Josh Klontz
* **properties:**

Property | Type | Description
--- | --- | ---
detectOnly | bool | If true, return all detected faces. Otherwise, return only faces that are suitable for recognition. Default is false.

---

# PP5CompareDistance

Compare templates with PP5

* **file:** classification/pp5.cpp
* **inherits:** [UntrainableDistance](../cpp_api/untrainabledistance/untrainabledistance.md)
* **authors:** Josh Klontz, E. Taborsky
* **properties:** None


---

# PP5EnrollTransform

Enroll faces in PP5

* **file:** classification/pp5.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **authors:** Josh Klontz, E. Taborsky
* **properties:**

Property | Type | Description
--- | --- | ---
detectOnly | bool | If true, enroll all detected faces. Otherwise, only enroll faces suitable for recognition. Default is false.
requireLandmarks | bool | If true, require the right eye, left eye, and nose base to be detectable by PP5. If this does not happen FTE is set to true for that template. Default is false.
adaptiveMinSize | float | The minimum face size as a percentage of total image width. 0.1 corresponds to a minimum face size of 10% the total image width. Default is 0.01.
minSize | int | The absolute minimum face size to search for. This is not a pixel value. Please see PittPatt documentation for the relationship between minSize and pixel IPD. Default is 4.
landmarkRange | enum | Range of landmarks to search for. Options are Frontal, Extended, Full, and Comprehensive. Default is Comprehensive.
searchPruningAggressiveness | int | The amount of aggressiveness involved in search for faces in images. 0 means all scales and locations are searched. 1 means fewer detectors are used in the early stages but all scales are still searched. 2-4 means that the largest faces are found first and then fewer scales are searched. Default is 0.

---

# RowWisePCATransform

PCA on each row.

* **file:** classification/lda.cpp
* **inherits:** [PCATransform](#pcatransform)
* **author:** Josh Klontz
* **properties:** None


---

# SVMTransform

Wraps OpenCV's SVM framework.

* **file:** classification/svm.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **see:**

	* [http://docs.opencv.org/modules/ml/doc/support_vector_machines.html](http://docs.opencv.org/modules/ml/doc/support_vector_machines.html)
	* [C. Burges. "A tutorial on support vector machines for pattern recognition", Knowledge Discovery and Data Mining 2(2), 1998.](C. Burges. "A tutorial on support vector machines for pattern recognition", Knowledge Discovery and Data Mining 2(2), 1998.)

* **author:** Josh Klontz
* **properties:**

Property | Type | Description
--- | --- | ---
Kernel | enum | The type of SVM kernel to use. Options are Linear, Poly, RBF, Sigmoid. Default is Linear.
Type | enum | The type of SVM to do. Options are C_SVC, NU_SVC, ONE_CLASS, EPS_SVR, NU_SVR. Default is C_SVC.
C | float | Parameter C of an SVM optimization problem. Needed when Type is C_SVC, EPS_SVR or NU_SVR. Default is -1.
gamma | float | Parameter gamma of a kernel function. Needed when Kernel is Poly, RBF, or Sigmoid. Default is -1.
inputVariable | QString | Metadata variable storing the label for each template. Default is "Label".
outputVariable | QString | Metadata variable to store the prediction value of the trained SVM. If type is EPS_SVR or NU_SVR the stored value is the output of the SVM. Otherwise the value is the output of the SVM mapped through the reverse lookup table. Default is "".
returnDFVal | bool | If true, dst is set to a 1x1 Mat with value equal to the predicted output of the SVM. Default is false.
termCriteria | int | The maximum number of training iterations. Default is 1000.
folds | int | Cross validation parameter used for autoselecting other parameters. Default is 5.
balanceFolds | bool | If true and the problem is 2-class classification then more balanced cross validation subsets are created. Default is false.

---

# SparseLDATransform

Projects input into learned Linear Discriminant Analysis subspace learned on a sparse subset of features with the highest weight in the original LDA algorithm.

* **file:** classification/lda.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author:** Brendan Klare
* **properties:**

Property | Type | Description
--- | --- | ---
varThreshold | float | BRENDAN FILL ME IN. Default is 1.5.
pcaKeep | float | BRENDAN FILL ME IN. Default is 0.98.
normalize | bool | BRENDAN FILL ME IN. Default is true.

---

# TurkClassifierTransform

Convenience class for training turk attribute regressors

* **file:** classification/turk.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author:** Josh Klontz
* **properties:**

Property | Type | Description
--- | --- | ---
key | QString | Metadata key to pass input values to SVM. Actual lookup key is "key_value" where value is each value in the parameter values. Default is "".
values | QStringList | Metadata keys to pass input values to SVM. Actual lookup key is "key_value" where key is the parameter key and value is each value in this list. Each passed value trains a new SVM with the input values found in metadata<ul><li>"key_value"</li></ul>. Default is "".
isMeta | bool | If true, "Average+SaveMat(predicted_key_value)" is appended to each classifier. If false, nothing is appended. Default is false.

---

