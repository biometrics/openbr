<!-- OUTPUT -->

Inherits from [Object](#object)

## Properties {: #output-properties }

Property | Type | Description
--- | --- | ---
<a class="table-anchor" id=output-properties-blockrows></a>blockRows | int | DOCUMENT ME
<a class="table-anchor" id=output-properties-blockcols></a>blockCols | int | DOCUMENT ME

## Members {: #output-members }

Member | Type | Description
--- | --- | ---
<a class="table-anchor" id=output-members-targetfiles></a>targetFiles | [FileList][#filelist] | DOCUMENT ME
<a class="table-anchor" id=output-members-queryfiles></a>queryFiles | [FileList](#filelist) | DOCUMENT ME
<a class="table-anchor" id=output-members-selfsimilar></a>selfSimilar | bool | DOCUMENT ME
<a class="table-anchor" id=output-members-next></a>next | [QSharedPointer][QSharedPointer]<[Output](#output)> | DOCUMENT ME
<a class="table-anchor" id=output-members-offset></a>offset | [QPoint][QPoint] | DOCUMENT ME

## Constructors {: #output-constructors }

Constructor \| Destructor | Description
--- | ---
virtual ~Output() | DOCUMENT ME

## Static Functions {: #output-static-functions }

### Output \*make(const [File][#file] &file, const [FileList](#filelist) &targetFiles, const [FileList](#filelist) &queryFiles) {: #output-function-make}

DOCUMENT ME

* **function definition:**

		static Output *make(const File &file, const FileList &targetFiles, const FileList &queryFiles)

* **parameters:**

	Parameter | Type | Description
	--- | --- | ---
	file | const [File](#file) & | DOCUMENT ME
	targetFiles | const [FileList](#filelist) & | DOCUMENT ME
	queryFiles | const [FileList](#filelist) & | DOCUMENT ME

* **output:** ([Output](#output)) DOCUMENT ME


## Functions {: #output-functions }

### virtual void initialize(const [FileList](#filelist) &targetFiles, const [FileList](#filelist) &queryFiles) {: #output-function-initialize }

DOCUMENT ME

* **function definition:**

		virtual void initialize(const [FileList](#filelist) &targetFiles, const [FileList](#filelist) &queryFiles)

* **parameters:**

	Parameter | Type | Description
	--- | --- | ---
	targetFiles | const [FileList](#filelist) & | DOCUMENT ME
	queryFiles | const [FileList](#filelist) & | DOCUMENT ME

* **output:** (void) DOCUMENT ME


### virtual void setBlock(int rowBlock, int columnBlock) {: #output-function-setblock }

DOCUMENT ME

* **function definition:**

		virtual void setBlock(int rowBlock, int columnBlock)

* **parameters:**

	Parameter | Type | Description
	--- | --- | ---
	rowBlock | int | DOCUMENT ME
	columnBlock | int | DOCUMENT ME

* **output:** (void) DOCUMENT ME


### virtual void setRelative(float value, int i, int j) {: #output-function-setrelative }

DOCUMENT ME

* **function definition:**

		virtual void setRelative(float value, int i, int j)

* **parameters:**

	Parameter | Type | Description
	--- | --- | ---
	value | float | DOCUMENT ME
	i | int | DOCUMENT ME
	j | int | DOCUMENT ME

* **output:** (void) DOCUMENT ME


### virtual void set(float value, int i, int j) = 0 {: #output-function-set }

DOCUMENT ME

* **function definition:**

		virtual void set(float value, int i, int j) = 0

* **parameters:**

	Parameter | Type | Description
	--- | --- | ---
	value | float | DOCUMENT ME
	i | int | DOCUMENT ME
	j | int | DOCUMENT ME

* **output:** (void) DOCUMENT ME
