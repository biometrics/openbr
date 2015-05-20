## [QString][QString] toString(int row, int column) {: #tostring}

Get a value in [data](members.md#data) as a string using a provided row and column index.

* **function definition:**

        QString toString(int row, int column) const

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    row | int | Row index of value
    column | int | Column index of value

* **output:** ([QString][QString]) Returns the value stored at (row, column) as a string
* **example:**

        TemplateList targets = TemplateList() << Template("target1.jpg") << Template("target2.jpg") << Template("target3.jpg");
        TemplateList queries = TemplateList() << Template("query1.jpg") << Template("query2.jpg");

        MatrixOutput *output = MatrixOutput::make(targets, queries);
        output->set(10.0, 1, 2);
        output->toString(1, 2); // Returns "10"
        output->toString(2, 2); // ERROR: row index is out of range

## void initialize(const [FileList](../filelist/filelist.md) &targetFiles, const [FileList](../filelist/filelist.md) &queryFiles) {: #initialize}

Initialize the output. This function calls [initialize](../output/functions.md#initialize) which should be overloaded by derived classes that need to be initialized. After calling [initialize](../output/functions.md#initialize), [data](members.md#data) is initialized to be of size queryFiles.size() x targetFiles.size().

* **function definition:**

        void initialize(const FileList &targetFiles, const FileList &queryFiles)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    targetFiles | const [FileList](../filelist/filelist.md) & | List of target files for initialization
    queryFiles | const [FileList](../filelist/filelist.md) & | List of query files for initialization

* **output:** (void)


## void set(float value, int i, int j) {: #set}

Set a value in [data](members.md#data) at the provided row and column indices.

* **function definition:**

        void set(float value, int i, int j)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    value | float | Value to be set
    i | int | Row index into [data](members.md#data)
    j | int | Column index into [data](members.md#data)

* **output:** (void)
* **example:**

        TemplateList targets = TemplateList() << Template("target1.jpg") << Template("target2.jpg") << Template("target3.jpg");
        TemplateList queries = TemplateList() << Template("query1.jpg") << Template("query2.jpg");

        MatrixOutput *output = MatrixOutput::make(targets, queries);
        output->set(6.0, 0, 1);
        output->toString(0, 1); // Returns "6.0"

        output->set(10.0, 1, 2);
        output->toString(1, 2); // Returns "10.0"

<!-- Links -->
[QString]: http://doc.qt.io/qt-5/QString.html "QString"
