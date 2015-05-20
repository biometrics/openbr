## void train(const [QList][QList]&lt;[Mat][Mat]&gt; &images, const [QList][QList]&lt;float&gt; &labels) {: #train }

This is a pure, virtual function. Train the classifier using the provided images and labels.

* **function definition:**

        virtual void train(const QList<Mat> &images, const QList<float> &labels) = 0

* **parameters:**

    Parameter | Type | Descriptions
    --- | --- | ---
    images | const [QList][QList]&lt;[Mat][Mat]&gt; & | Training images
    labels | const [QList][QList]&lt;float&gt; & | Training labels

* **output:** (void)
*  **example:**

        // Create data for a 2-class classification problem
        QList<Mat> images = QList<Mat>() << Template("training_pic1.jpg").m()
                                         << Template("training_pic2.jpg").m()
                                         << Template("training_pic3.jpg").m()
                                         << Template("training_pic4.jpg").m();

        QList<float> labels = QList<float>() << 0 << 0 << 1 << 1;

        Classifier *classifier = Classifier::make("Classifier");
        rep->train(images, labels);

## float classify(const [Mat][Mat] &image) const {: #classify }

This is a pure virtual function. Classify a provided input image.

* **function description:**

        virtual float classify(const Mat &image) const = 0

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    image | const [Mat][Mat] & | Input image to be classified

* **output:** (float) Returns the classification value of the image. The value can be a confidence, a regression, or a class. In 2-class classification is it often a confidence which has been normalized such that 0 is the inflection point. Values below zero represent a negative classification and values above represent a positive classification.
* **example:**
        Classifier *classifier = Classifier::make("2ClassClassifier"); // assume classifier is already trained

        Template p1("pos_image1.jpg"); // positive sample
        Template n1("neg_image1.jpg"); // negative sample

        classifier->classify(p1); // returns confidence > 0
        classifier->classify(n1); // returns confidence < 0

<!-- Links -->
[QList]: http://doc.qt.io/qt-5/QList.html "QList"
[Mat]: http://docs.opencv.org/modules/core/doc/basic_structures.html#mat "Mat"
