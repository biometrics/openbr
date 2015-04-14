<!-- MATRIX OUTPUT -->

Inherits from [Output](#output)

## Properties {: #matrixoutput-properties }

NONE

## Members {: #matrixoutput-members }

Member | Type | Description
--- | --- | ---
<a class="table-anchor" id=matrixoutput-members-data></a>data | [Mat][Mat] | DOCUMENT ME

## Constructors {: #matrixoutput-constructors }

NONE

## Static Functions {: #matrixoutput-static-functions }

### static [MatrixOutput](#matrixoutput) \*make(const [FileList](#filelist) &targetFiles, const [FileList](#filelist) &queryFiles) {: #matrixoutput-function-make}

DOCUMENT ME

* **function definition:**

		static MatrixOutput *make(const FileList &targetFiles, const FileList &queryFiles)

* **parameters:**

	Parameter | Type | Description
	--- | --- | ---
	targetFiles | const [FileList](#filelist) & | DOCUMENT ME
	queryFiles | const [FileList](#filelist) & | DOCUMENT ME

* **output:** ([MatrixOutput](#matrixoutput)) DOCUMENT ME


## Functions {: #matrixoutput-functions }

### [QString][QString] toString(int row, int column) const {: #matrixoutput-function-tostring}

DOCUMENT ME

* **function definition:**

		QString toString(int row, int column) const

* **parameters:**

	Parameter | Type | Description
	--- | --- | ---
	row | int | DOCUMENT ME
	column | int | DOCUMENT ME

* **output:** ([QString][QString]) DOCUMENT ME


### void initialize(const [FileList](#filelist) &targetFiles, const [FileList](#filelist) &queryFiles) {: #matrixoutput-function-initialize}

DOCUMENT ME

* **function definition:**

		void initialize(const FileList &targetFiles, const FileList &queryFiles)

* **parameters:**

	Parameter | Type | Description
	--- | --- | ---
	targetFiles | const [FileList](#filelist) & | DOCUMENT ME
	queryFiles | const [FileList](#filelist) & | DOCUMENT ME

* **output:** (void) DOCUMENT ME


### void set(float value, int i, int j) {: #matrixoutput-function-set}

DOCUMENT ME

* **function definition:**

		void set(float value, int i, int j)

* **parameters:**

	Parameter | Type | Description
	--- | --- | ---
	value | float | DOCUMENT ME
	i | int | DOCUMENT ME
	j | int | DOCUMENT ME

* **output:** (void) DOCUMENT ME
