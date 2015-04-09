# CollectNNTransform

Collect nearest neighbors and append them to metadata.

* **file:** cluster/collectnn.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api.md#untrainablemetatransform)
* **author:** Charles Otto
* **properties:**

Property | Type | Description
--- | --- | ---
keep | int | The maximum number of nearest neighbors to keep. Default is 20.

---

# KMeansTransform

Wraps OpenCV kmeans and flann.

* **file:** cluster/kmeans.cpp
* **inherits:** [Transform](../cpp_api.md#transform)
* **see:** [http://docs.opencv.org/modules/flann/doc/flann_fast_approximate_nearest_neighbor_search.html](http://docs.opencv.org/modules/flann/doc/flann_fast_approximate_nearest_neighbor_search.html)
* **author:** Josh Klontz
* **properties:**

Property | Type | Description
--- | --- | ---
kTrain | int | The number of random centroids to make at train time. Default is 256.
kSearch | int | The number of nearest neighbors to search for at runtime. Default is 1.

---

# KNNTransform

K nearest neighbors classifier.

* **file:** cluster/knn.cpp
* **inherits:** [Transform](../cpp_api.md#transform)
* **author:** Josh Klontz
* **properties:** None


---

# LogNNTransform

Log nearest neighbors to specified file.

* **file:** cluster/lognn.cpp
* **inherits:** [TimeVaryingTransform](../cpp_api.md#timevaryingtransform)
* **author:** Charles Otto
* **properties:**

Property | Type | Description
--- | --- | ---
fileName | QString | The name of the log file. An empty fileName won't be written to. Default is "".

---

# RandomCentroidsTransform

Chooses k random points to be centroids.

* **file:** cluster/randomcentroids.cpp
* **inherits:** [Transform](../cpp_api.md#transform)
* **see:** [http://docs.opencv.org/modules/flann/doc/flann_fast_approximate_nearest_neighbor_search.html](http://docs.opencv.org/modules/flann/doc/flann_fast_approximate_nearest_neighbor_search.html)
* **author:** Austin Blanton
* **properties:**

Property | Type | Description
--- | --- | ---
kTrain | int | The number of random centroids to make at train time. Default is 256.
kSearch | int | The number of nearest neighbors to search for at runtime. Default is 1.

---

