## static [MatrixOutput](matrixoutput.md) \*make(const [FileList](../filelist/filelist.md) &targetFiles, const [FileList](../filelist/filelist.md) &queryFiles) {: #make}

Make an [MatrixOutput](matrixoutput.md) from lists of target and query files. This function calls [Output](../output/output.md)::[make](../output/statics.md#make) using the string "Matrix". [Output](../output/output.md)::[make](../output/statics.md#make) in turn calls [initialize](functions.md#initialize). Initialize calls [initialize](../output/functions.md#initialize) which should be overloaded by derived classes to handle initialization.

* **function definition:**

        static MatrixOutput *make(const FileList &targetFiles, const FileList &queryFiles)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    targetFiles | const [FileList](../filelist/filelist.md) & | List of files representing the target templates
    queryFiles | const [FileList](../filelist/filelist.md) & | List of files representing the query templates

* **output:** ([MatrixOutput](matrixoutput.md) \*) Returns a pointer to the first output in the linked list
* **example:**

        TemplateList targets = TemplateList() << Template("target1.jpg") << Template("target2.jpg") << Template("target3.jpg");
        TemplateList queries = TemplateList() << Template("query1.jpg") << Template("query2.jpg");

        MatrixOutput *output = MatrixOutput::make(targets, queries); // returns a pointer to a MatrixOutput
