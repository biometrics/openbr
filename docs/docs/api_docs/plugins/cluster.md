# CollectNNTransform

Collect nearest neighbors and append them to metadata.
 

* **file:** cluster/collectnn.cpp
* **inherits:** [UntrainableMetaTransform](../cpp_api/untrainablemetatransform/untrainablemetatransform.md)
* **author(s):** [Charles Otto][caotto]
* **properties:**

	Property | Type | Description
	--- | --- | ---
	keep | int | The maximum number of nearest neighbors to keep. Default is 20.

---

# KMeansTransform

Wraps OpenCV kmeans and flann.
 

* **file:** cluster/kmeans.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author(s):** [Josh Klontz][jklontz]
* **see:** [http://docs.opencv.org/modules/flann/doc/flann_fast_approximate_nearest_neighbor_search.html](http://docs.opencv.org/modules/flann/doc/flann_fast_approximate_nearest_neighbor_search.html)
* **properties:**

	Property | Type | Description
	--- | --- | ---
	kTrain | int | The number of random centroids to make at train time. Default is 256.
	kSearch | int | The number of nearest neighbors to search for at runtime. Default is 1.

---

# KNNTransform

K nearest neighbors classifier.
 

* **file:** cluster/knn.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author(s):** [Josh Klontz][jklontz]
* **properties:** None


---

# LogNNTransform

Log nearest neighbors to specified file.
 

* **file:** cluster/lognn.cpp
* **inherits:** [TimeVaryingTransform](../cpp_api/timevaryingtransform/timevaryingtransform.md)
* **author(s):** [Charles Otto][caotto]
* **properties:**

	Property | Type | Description
	--- | --- | ---
	fileName | QString | The name of the log file. An empty fileName won't be written to. Default is "".

---

# RandomCentroidsTransform

Chooses k random points to be centroids.
 

* **file:** cluster/randomcentroids.cpp
* **inherits:** [Transform](../cpp_api/transform/transform.md)
* **author(s):** [Austin Blanton][imaus10]
* **see:** [http://docs.opencv.org/modules/flann/doc/flann_fast_approximate_nearest_neighbor_search.html](http://docs.opencv.org/modules/flann/doc/flann_fast_approximate_nearest_neighbor_search.html)
* **properties:**

	Property | Type | Description
	--- | --- | ---
	kTrain | int | The number of random centroids to make at train time. Default is 256.
	kSearch | int | The number of nearest neighbors to search for at runtime. Default is 1.

---

