<!-- API FUNCTIONS -->

## IsClassifier {: #isclassifier }

Determines if the given algorithm is a classifier. A classifier is defined as a [Transform](transform/transform.md) with no associated [Distance](distance/distance.md). Instead metadata fields with the predicted output classes are populated in [Template](template/template.md) [files](template/members.md#file).

* **function definition:**

        bool IsClassifier(const QString &algorithm)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    algorithm | const [QString][QString] & | Algorithm to evaluate

* **output:** (bool) True if the algorithm is a classifier and false otherwise
* **see:** [br_is_classifier](../c_api/functions.md#br_is_classifier)
* **example:**

        IsClassifier("Identity"); // returns true
        IsClassifier("Identity:Dist"); // returns false

---

## Train {: #train }

High level function for creating models.

* **function definition:**

        void Train(const File &input, const File &model)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    input | const [File](file/file.md) & | Training data
    model | const [File](file/file.md) & | Model file

* **output:** (void)
* **see:** The [training tutorial](../../tutorials.md#training-algorithms) for an example of training.
* **example:**

        File file("/path/to/images/or/gallery.gal");
        File model("/path/to/model/file");
        Train(file, model);

---

## Enroll {: #enroll-1 }

High level function for creating [galleries](gallery/gallery.md).

* **function definition:**

        void Enroll(const File &input, const File &gallery = File())

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    input | const [File](file/file.md) & | Path to enrollment file
    gallery | const [File](file/file.md) & | (Optional) Path to gallery file.

* **output:** (void)
* **see:** [br_enroll](../c_api/functions.md#br_enroll)
* **example:**

        File file("/path/to/images/or/gallery.gal");
        Enroll(file); // Don't need to specify a gallery file
        File gallery("/path/to/gallery/file");
        Enroll(file, gallery); // Will write to the specified gallery file

---

## Enroll {: #enroll-2 }

High level function for enrolling templates. Templates are modified in place as they are projected through the algorithm.

* **function definition:**

        void Enroll(TemplateList &tmpl)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    tmpl | [TemplateList](templatelist/templatelist.md) & | Data to enroll

* **output:** (void)
* **example:**

        TemplateList tList = TemplateList() << Template("picture1.jpg")
                                            << Template("picture2.jpg")
                                            << Template("picture3.jpg");
        Enroll(tList);

---

## Project {: #project}

A naive alternative to [Enroll](#enroll-1).

* **function definition:**

        void Project(const File &input, const File &output)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    input | const [File](file/file.md) & | Path to enrollment file
    gallery | const [File](file/file.md) & | Path to gallery file.

* **output:** (void)
* **see:** [Enroll](#enroll-1)
* **example:**

        File file("/path/to/images/or/gallery.gal");
        File output("/path/to/gallery/file");
        Project(file, gallery); // Will write to the specified gallery file

---

## Compare {: #compare }

High level function for comparing galleries. Each template in the **queryGallery** is compared against every template in the **targetGallery**.

* **function definition:**

        void Compare(const File &targetGallery, const File &queryGallery, const File &output)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    targetGallery | const [File](file/file.md) & | Gallery of target templates
    queryGallery | const [File](file/file.md) & | Gallery of query templates
    output | const [File](file/file.md) & | Output file for results

* **returns:** (output)
* **see:** [br_compare](../c_api/functions.md#br_compare)
* **example:**

        File target("/path/to/target/images/");
        File query("/path/to/query/images/");
        File output("/path/to/output/file");
        Compare(target, query, output);

---

## CompareTemplateList {: #comparetemplatelists}

High level function for comparing templates.

* **function definition:**

        void CompareTemplateLists(const TemplateList &target, const TemplateList &query, Output *output);

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    target | const [TemplateList](templatelist/templatelist.md) & | Target templates
    query | const [TemplateList](templatelist/templatelist.md) & | Query templates
    output | [Output](output/output.md) \* | Output file for results

* **output:** (void)
* **example:**

        TemplateList targets = TemplateList() << Template("target_img1.jpg")
                                              << Template("target_img2.jpg")
                                              << Template("target_img3.jpg");

        TemplateList query = TemplateList() << Template("query_img.jpg");
        Output *output = Factory::make<Output>("/path/to/output/file");

        CompareTemplateLists(targets, query, output);

---

## PairwiseCompare {: #pairwisecompare }

High level function for doing a series of pairwise comparisons.

* **function definition:**

        void PairwiseCompare(const File &targetGallery, const File &queryGallery, const File &output)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    targetGallery | const [File](file/file.md) & | Gallery of target templates
    queryGallery | const [File](file/file.md) & | Gallery of query templates
    output | const [File](file/file.md) & | Output file for results  

* **output:** (void)
* **see:** [br_pairwise_comparison](../c_api/functions.md#br_pairwise_compare)
* **example:**

        File target("/path/to/target/images/");
        File query("/path/to/query/images/");
        File output("/path/to/output/file");
        PairwiseCompare(target, query, output);

---

## Convert {: #convert }

Change the format of the **inputFile** to the format of the **outputFile**. Both the **inputFile** format and the **outputFile** format must be of the same format group, which is specified by the **fileType**.

* **function definition:**

        void Convert(const File &fileType, const File &inputFile, const File &outputFile)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    fileType | const [File](file/file.md) & | Can be either: <ul> <li>[Format](format/format.md)</li> <li>[Gallery](gallery/gallery.md)</li> <li>[Output](output/output.md)</li> </ul>
    inputFile | const [File](file/file.md) & | File to be converted. Format is inferred from the extension.
    outputFile | const [File](file/file.md) & | File to store converted input. Format is inferred from the extension.

* **output:** (void)
* **example:**

        File input("input.csv");
        File output("output.xml");
        Convert("Format", input, output);

---

## Cat {: #cat }

Concatenate several galleries into one.

* **function definition:**

        void Cat(const QStringList &inputGalleries, const QString &outputGallery)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    inputGalleries | const [QStringList][QStringList] & | List of galleries to concatenate
    outputGallery | const [QString][QString] & | Gallery to store the concatenated result. This gallery cannot be in the inputGalleries

* **output:** (void)
* **see:** [br_cat](../c_api/functions.md#br_cat)
* **example:**

        QStringList inputGalleries = QStringList() << "/path/to/gallery1"
                                                   << "/path/to/gallery2"
                                                   << "/path/to/gallery3";

        QString outputGallery = "/path/to/outputGallery";
        Cat(inputGalleries, outputGallery);

---

## Deduplicate {: #deduplicate }

Deduplicate a gallery. A duplicate is defined as an image with a match score above a given threshold to another image in the gallery.

* **function definition:**

        void Deduplicate(const File &inputGallery, const File &outputGallery, const QString &threshold)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    inputGallery | const [File](file/file.md) & | Gallery to deduplicate
    outputGallery | const [File](file/file.md) & | Gallery to store the deduplicated result
    threshold | const [QString][QString] & | Match score threshold to determine duplicates

* **output:** (void)
* **see:** [br_deduplicate](../c_api/functions.md#br_deduplicate)
* **example:**

        File input("/path/to/input/galley/with/dups");
        File output("/path/to/output/gallery");
        Deduplicate(input, output, "0.7"); // Remove duplicates with match scores above 0.7

<!-- Links -->
[QString]: http://doc.qt.io/qt-5/QString.html "QString"
[QStringList]: http://doc.qt.io/qt-5/qstringlist.html "QStringList"
