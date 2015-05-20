## [Mat][Mat] preprocess(const [Mat][Mat] &image) {: #preprocess }

This is a virtual function. Preprocess an image into the desired format for the representation. Default implementation returns the image unmodified.

* **function definition:**

        virtual Mat preprocess(const Mat &image) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    image | const [Mat][Mat] & | Image to be preprocessed

* **output:** ([Mat][Mat]) Returns the preprocessed image
* **example:**

        Template in("picture.jpg");

        Representation *rep = Representation::make("RepresentationThatRequiresGrayscale");
        rep->preprocess(in); // returns the original image converted to grayscale

## void train(const [QList][QList]&lt;[Mat][Mat]&gt; &images, const [QList][QList]&lt;float&gt; &labels) {: #train }

This is a virtual function. Train the representation using the provided images and associated labels. Default implementation does no training.

* **function definition:**

        virtual void train(const QList<Mat> &images, const QList<float> &labels)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    images | const [QList][QList]&lt;[Mat][Mat]&gt; & | Training images
    labels | const [QList][QList]&lt;float&gt; & | Training labels

* **output:** (void)
* **example:**

        // Create data for a 2-class classification problem
        QList<Mat> images = QList<Mat>() << Template("training_pic1.jpg").m()
                                         << Template("training_pic2.jpg").m()
                                         << Template("training_pic3.jpg").m()
                                         << Template("training_pic4.jpg").m();

        QList<float> labels = QList<float>() << 0 << 0 << 1 << 1;

        Representation *rep = Representation::make("Representation");
        rep->train(images, labels);

## [Mat][Mat] evaluate(const [Mat][Mat] &image, const [QList][QList]&lt;int&gt; &indices = [QList][QList]&lt;int&gt;()) {: #evaluate }

This is a pure virtual function. For a provided input image calculate only the feature responses associated with the provided indices. The indices are expected relative to the entire feature space. If the indices list is empty the response of the entire feature space is calculated.

* **function definition:**

        virtual cv::Mat evaluate(const Mat &image, const QList<int> &indices = QList<int>()) const = 0

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    image | const [Mat][Mat] & | The image to be converted
    indices | const [QList][QList]&lt;int&gt; & | (Optional) A list of indices corresponding to the desired features to calculate. If the list is empty all features are calculated.

* **output:** ([Mat][Mat]) Returns a 1xN feature vector where N is the number of indices provided. If no indices are provided N equals the size of the feature space.
* **example:**

        Template image("picture.jpg");

        Representation *rep = Representation::make("Representation");
        rep->evaluate(image, QList<int>() << 7 << 10 << 72 ); // returns a 1x3 Mat feature vector

## int numFeatures() {: #numfeatures }

This is a pure virtual function. Get the size of the feature space.

* **function definition:**

        virtual int numFeatures() const = 0

* **parameters:** NONE
* **output:** (int) Returns the size of the feature space
* **example:**

        Representation *rep1 = Representation::make("RepresentationWith1000features");
        Representation *rep2 = Representation::make("RepresentationWith25643features");

        rep1->numFeatures(); // returns 1000
        rep2->numFeatures(); // returns 25643

<!-- Links -->
[QList]: http://doc.qt.io/qt-5/QList.html "QList"
[Mat]: http://docs.opencv.org/modules/core/doc/basic_structures.html#mat "Mat"
