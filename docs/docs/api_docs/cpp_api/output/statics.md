## Output \*make(const [File](../file/file.md) &file, const [FileList](../filelist/filelist.md) &targetFiles, const [FileList](../filelist/filelist.md) &queryFiles) {: #make}

Make an [Output](output.md) from a string and lists of target and query files. This function calls [initialize](functions.md#initialize), which should be overloaded by derived classes to handle initialization. The provided file is first split using [File](../file/file.md)::[split](../file/functions.md#split-1) and each resulting file is turned into an [Output](output.md) that is stored in a linked-list.

* **function definition:**

        static Output *make(const File &file, const FileList &targetFiles, const FileList &queryFiles)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    file | const [File](../file/file.md) & | File describing the output or outputs to construct
    targetFiles | const [FileList](../filelist/filelist.md) & | List of files representing the target templates
    queryFiles | const [FileList](../filelist/filelist.md) & | List of files representing the query templates

* **output:** ([Output](output.md) \*) Returns a pointer to the first output in the linked list
* **example:**

        TemplateList targets = TemplateList() << Template("target1.jpg") << Template("target2.jpg") << Template("target3.jpg");
        TemplateList queries = TemplateList() << Template("query1.jpg") << Template("query2.jpg");

        Output *output1 = Output::make("output.mtx", targets, queries); // returns a pointer to an Output at "output.mtx"
        Output *output2 = Output::make("output1.mtx;output2.mtx", targets, queries); // returns a pointer to the output created with "output1.mtx" with a pointer to the output created with "output2.mtx"
