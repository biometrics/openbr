## void initialize(const [FileList](../filelist/filelist.md) &targetFiles, const [FileList](../filelist/filelist.md) &queryFiles) {: #initialize }

This is a virtual function. Initialize the output with provided target and query files.

* **function definition:**

        virtual void initialize(const [FileList](../filelist/filelist.md) &targetFiles, const [FileList](../filelist/filelist.md) &queryFiles)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    targetFiles | const [FileList](../filelist/filelist.md) & | Target files to initialize the Output with
    queryFiles | const [FileList](../filelist/filelist.md) & | Query files to initialize the Output with

* **output:** (void)
* **example:**

        TemplateList targets = TemplateList() << Template("target1.jpg") << Template("target2.jpg") << Template("target3.jpg");
        TemplateList queries = TemplateList() << Template("query1.jpg") << Template("query2.jpg");

        Output *output = Factory::make<Output>("output.mtx");
        output->initialize(targets, queries); // This is the same as calling Output::make("output.mtx", targets, queries)

## void setBlock(int rowBlock, int columnBlock) {: #setblock }

This is a virtual function. Set the read offset of the Output.

* **function definition:**

        virtual void setBlock(int rowBlock, int columnBlock)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    rowBlock | int | Row position of the offset
    columnBlock | int | Column position of the offset

* **output:** (void)

## void setRelative(float value, int i, int j) {: #setrelative }

This is a virtual function. Set a value in the Output. **i** and **j** are *relative* to the current block.

* **function definition:**

        virtual void setRelative(float value, int i, int j)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    value | float | Value to set in the output
    i | int | Row value relative to the current block
    j | int | Column value relative to the current block

* **output:** (void)


## void set(float value, int i, int j) {: #set }

This is a pure virtual function. Set a value in the output.

* **function definition:**

        virtual void set(float value, int i, int j) = 0

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    value | float | Value to be inserted into the output
    i | int | Row index to insert at
    j | int | Column index to insert at

* **output:** (void)
