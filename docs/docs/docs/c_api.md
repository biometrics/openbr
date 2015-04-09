The C API is a high-level API for running algorithms and evaluating results.

In order to provide a high-level interface that is usable from the command line and callable from other programming languages, the API is designed to operate at the "file system" level.
In other words, arguments to many functions are file paths that specify either a source of input or a desired output.
File extensions are relied upon to determine *how* files should be interpreted in the context of the function being called.
The [C++ Plugin API](cpp_api.md) should be used if more fine-grained control is required.

Import API considerations include-

* Memory for <tt>const char*</tt> return values is managed internally and guaranteed until the next call to the function
* Users should input a char * buffer and the size of that buffer. String data will be copied into the buffer, if the buffer is too small, only part of the string will be copied. Returns the buffer size required to contain the complete string.

To use the API in your project use-

    #include <openbr/openbr.h>

[CMake](http://www.cmake.org/) developers may wish to the cmake configuration file found at

    share/openbr/cmake/OpenBRConfig.cmake

Please see the [tutorials](../tutorials.md) section for examples.

---

# Typedefs

## void *br_template

## void *br_template_list

## void *br_gallery

## void *br_matrix_output

---

# Functions

## br_about

Wraps [Context](cpp_api.md#context)

* **return type:** const char *
* **parameters:** None
* **example call:** ```const char *br_about()```
* **see:** [br_version](#br_version)

---

## br_cat

Concatenates a list of galleries into 1 gallery.

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
 num_input_galleries | int | Parameter description
input_galleries[] | const char * | Parameter description
output_gallery | const char * | Parameter description

* **example call:** ```void br_cat(int num_input_galleries, const char *input_galleries[], const char *output_gallery)```

---

## br_deduplicate

Removes duplicate [templates](cpp_api.md#template) in a [gallery](cpp_api.md#gallery). If a galley contains n duplicates, the first n-1 duplicates in the gallery will be removed and the nth will be kept. Users are encouraged to use binary gallery formats as the entire gallery is read into memory in one call to [Gallery](cpp_api.md#gallery)::read.

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
input_gallery | const char * | Gallery to be deduplicated
output_gallery | const char * | Deduplicated gallery
threshold | const char * | Comparisons with a match score >= this value are designated to be duplicates.

* **example call:** ```void br_deduplicate(const char *input_gallery, const char *output_gallery, const char *threshold)```

---

## br_cluster

\brief Clusters one or more similarity matrices into a list of subjects. A [similarity matrix](../technical.md#the-evaluation-harness) is a type of [Output](cpp_api.md#output). The current clustering algorithm is a simplified implementation of \cite zhu11.

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
num_simmats | int | Size of **simmats**
simmats[] | const char * | Array of [simmat](../technical.md#the-evaluation-harness) composing one large self-similarity matrix arranged in row major order.
aggressiveness | float | The higher the aggressiveness the larger the clusters. Suggested range is [0,10]
csv | const char * | The cluster results file to generate. Results are stored one row per cluster and use gallery indices.

* **example call:** ```void br_cluster(int num_simmats, const char *simmats[], float aggressiveness, const char *csv)```

---

## br_combine_masks

Combines several equal-sized mask matrices. A comparison may not be simultaneously indentified as both a genuine and an imposter by different input masks.

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
num_input_masks | int | Size of **input_masks**
input_masks[] | const char * | Array of [mask matrices](../technical.md#the-evaluation-harness) to combine. All matrices must have the same dimensions.
output_mask | const char * | The file to contain the resulting [mask matrix](../technical.md#the-evaluation-harness)
method | const char * | Possible values are: <ul><li>And - Ignore comparison if *any* input masks ignore.</li> <li>Or - Ignore comparison if *all* input masks ignore.</li></ul>

* **example call:** ```void br_combine_masks(int num_input_masks, const char *input_masks[], const char *output_mask, const char *method)```
* **see:** [br_make_mask](#br_make_mask)

---

## br_compare

Compares each template in the query gallery to each template in the target gallery.

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
target_gallery | const char * | target_gallery The br::Gallery file whose templates make up the columns of the output.
query_gallery | const char * | The br::Gallery file whose templates make up the rows of the output. A value of '.' reuses the target gallery as the query gallery.
output | const char * | (Optional) [Output](cpp_api.md#output) file to contain the results of comparing the templates. The default behavior is to print scores to the terminal.

* **example call:** ```void br_compare(const char *target_gallery, const char *query_gallery, const char *output = "")```
* **see:** br_enroll

---

## br_compare_n

Convenience function for comparing to multiple targets.

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
num_targets | int | Size of **target_galleries**
target_galleries[] | const char * | Target galleries to compare against
query_gallery | const char * | query gallery for comparison.
output | const char * | (Optional) [Output](cpp_api.md#output) file to contain the results of comparing the templates. The default behavior is to print scores to the terminal.

* **example call:** ```void br_compare_n(int num_targets, const char *target_galleries[], const char *query_gallery, const char *output)```
* **see:** br_compare

---

## br_pairwise_compare

DOCUMENT ME!

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
target_gallery | const char * | DOCUMENT ME
query_gallery | const char * | DOCUMENT ME
output | const char * | DOCUMENT ME

* **example call:** ```void br_pairwise_compare(const char *target_gallery, const char *query_gallery, const char *output = "")```

---

## br_convert

Convert a file to a different type. Files can only be converted to types within the same group. For example [formats](cpp_api.md#format) can only be converted to other [formats](cpp_api.md#format).

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
file_type | const char * | Type of file to convert. Options are Format, Gallery or Output.
input_file | const char * | File to convert.
output_file | const char * | Output file. Type is determined by the file extension.

* **example call:** ```void br_convert(const char *file_type, const char *input_file, const char *output_file)```

---

## br_enroll

Constructs template(s) from an input.

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
input | const char * | The [format](cpp_api.md#format) or [gallery](cpp_api.md#gallery) to enroll.
gallery | const char * | (Optional) The [Gallery](cpp_api.md#gallery) file to contain the enrolled templates. By default the gallery will be held in memory and *input* can used as a gallery in [br_compare](#br_compare)

* **example call:** ```void br_enroll(const char *input, const char *gallery = "")```
* **see:** [br_enroll_n](#br_enroll_n)

---

## br_enroll_n

Convenience function for enrolling multiple inputs.

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
num_inputs | int | Size of **inputs**.
inputs[] | const char * | Array of inputs to enroll.
gallery | const char * | (Optional) The [Gallery](cpp_api.md#gallery) file to contain the enroll templates.

* **example call:** ```void br_enroll_n(int num_inputs, const char *inputs[], const char *gallery = "")```
* **see:** [br_enroll](#br_enroll)

---

## br_project

A naive alternative to [br_enroll](#br_enroll).

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
input | const char * | DOCUMENT ME!
output | const char * | DOCUMENT ME!

* **example call:** ```void br_project(const char *input, const char *output)```
* **see:** [br_enroll](#br_enroll)

---

## br_eval

Creates a **.csv** file containing performance metrics from evaluating the similarity matrix using the mask matrix. The return value is the true accept rate at a false accept rate of one in one thousand.

* **return type:** float
* **parameters:**

Parameter | Type | Description
--- | --- | ---
simmat | const char * | The [simmat](../technical.md#the-evaluation-harness) to use
mask | const char * | The [mask](../technical.md#the-evaluation-harness) to use.
csv | const char * | (Optional) The **.csv** file to contain performance metrics.
matches | int | (Optional) An integer number of matches to output around the EER. Default is 0.

* **example call:** ```float br_eval(const char *simmat, const char *mask, const char *csv = "", int matches = 0)```
* **see:** [br_plot](#br_plot)

---

## br_assert_eval

Evaluates the similarity matrix using the mask matrix.  Function aborts ff TAR @ FAR = 0.001 does not meet an expected performance value.

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
simmat | const char * | The [simmat](../technical.md#the-evaluation-harness) to use
mask | const char * | The [mask](../technical.md#the-evaluation-harness)
accuracy | const float | Desired true accept rate at false accept rate of one in one thousand.

* **example call:** ```void br_assert_eval(const char *simmat, const char *mask, const float accuracy)```

---

## br_inplace_eval

Creates a **.csv** file containing performance metrics from evaluating the similarity matrix using galleries containing ground truth labels.

* **return type:** float
* **parameters:**

Parameter | Type | Description
--- | --- | ---
simmat | const char * | The [simmat](../technical.md#the-evaluation-harness)
target | const char * | The name of a gallery containing metadata for the target set.
query | const char * | The name of a gallery containing metadata for the query set.
csv | const char * | (Optional) The **.csv** file to contain performance metrics.

* **example call:** ```float br_inplace_eval(const char * simmat, const char *target, const char *query, const char *csv = "")```
* **see:** [br_plot](#br_plot)

---

## br_eval_classification

Evaluates and prints classification accuracy to terminal.

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
predicted_gallery | const char * | The predicted [Gallery](cpp_api.md#gallery).
truth_gallery | const char * | The ground truth [Gallery](cpp_api.md#gallery).
predicted_property | const char * | (Optional) Which metadata key to use from the **predicted_gallery**.
truth_property | const char * | (Optional) Which metadata key to use from the **truth_gallery**.

* **example call:** ```void br_eval_classification(const char *predicted_gallery, const char *truth_gallery, const char *predicted_property = "", const char *truth_property = "")```

---

## br_eval_clustering

Evaluates and prints clustering accuracy to the terminal.

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
csv | const char * | The cluster results file.
gallery | const char * | The [Gallery](cpp_api.md#gallery) used to generate the [simmat](../technical.md#the-evaluation-harness) that was clustered.
truth_property | const char * | (Optional) which metadata key to use from **gallery**, defaults to Label

* **example call:** ```void br_eval_clustering(const char *csv, const char *gallery, const char * truth_property)```

---

## br_eval_detection

Evaluates and prints detection accuracy to terminal.

* **return type:** float
* **parameters:**

Parameter | Type | Description
--- | --- | ---
predicted_gallery | const char * | The predicted [Gallery](cpp_api.md#gallery).
truth_gallery | const char * | The ground truth [Gallery](cpp_api.md#gallery).
csv | const char * | (Optional) The **.csv** file to contain performance metrics.
normalize | bool | (Optional) Flag to normalize predicted bounding boxes for improved detection. Defaults to false.
minSize | int | (Optional) Minimum size of faces to be considered in the evaluation. Size is applied to predicted and ground truth galleries. Defaults to -1 (no minimum size).
maxSize | int | (Optional) Maximum size if faces to be considered in the evaluation. Size is applied to predicted and ground truth galleries. Defaults to -1 (no maximum size).

* **example call:** ```float br_eval_detection(const char *predicted_gallery, const char *truth_gallery, const char *csv = "", bool normalize = false, int minSize = 0, int maxSize = 0)```

---

## br_eval_landmarking

Evaluates and prints landmarking accuracy to terminal.

* **return type:** float
* **parameters:**

Parameter | Type | Description
--- | --- | ---
predicted_gallery | const char * | The predicted [Gallery](cpp_api.md#gallery).
truth_gallery | const char * | The ground truth [Gallery](cpp_api.md#gallery).
csv | const char * | (Optional) The **.csv** file to contain performance metrics.
normalization_index_a | int | (Optional) The first index in the list of points to use for normalization. Default is 0.
normalization_index_b | int | (Optional) The second index in the list of points to use for normalization. Default is 1.
sample_index | int | (Optional) The index for sample landmark image in ground truth gallery. Default = 0.
total_examples | int | (Optional) The number of accurate and inaccurate examples to display. Default is 5.

* **example call:** ```float br_eval_landmarking(const char *predicted_gallery, const char *truth_gallery, const char *csv = "", int normalization_index_a = 0, int normalization_index_b = 1, int sample_index = 0, int total_examples = 5)```

---

## br_eval_regression

Evaluates regression accuracy to disk.

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
predicted_gallery | const char * | The predicted [Gallery](cpp_api.md#gallery)
truth_gallery | const char * | The ground truth [Gallery](cpp_api.md#gallery)
predicted_property | const char * | (Optional) Which metadata key to use from **predicted_gallery**.
truth_property | const char * | (Optional) Which metadata key to use from **truth_gallery**.

* **example call:** ```void br_eval_regression(const char *predicted_gallery, const char *truth_gallery, const char *predicted_property = "", const char *truth_property = "")```

---

## br_fuse

Perform score level fusion on similarity matrices.

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
num_input_simmats | int | Size of **input_simmats**.
input_simmats[] | const char * | Array of [simmats](../technical.md#the-evaluation-harness). All simmats must have the same dimensions.
normalization | const char * | Valid options are: <ul> <li>None - No score normalization.</li> <li>MinMax - Scores normalized to [0,1].</li> <li>ZScore - Scores normalized to a standard normal curve.</li> </ul>
fusion | const char * | Valid options are: <ul> <li>Min - Uses the minimum score.</li> <li>Max - Uses the maximum score.</li> <li>Sum - Sums the scores. Sums can also be weighted: <tt>SumW1:W2:...:Wn</tt>.</li> <li>Replace - Replaces scores in the first matrix with scores in the second matrix when the mask is set.</li> </ul>
output_simmat | const char * | [Simmat](../technical.md#the-evaluation-harness) to contain the fused scores.

* **example call:** ```void br_fuse(int num_input_simmats, const char *input_simmats[],
                       const char *normalization, const char *fusion, const char *output_simmat)```

---

## br_initialize

Initializes the [Context](cpp_api.md#context). Required at the beginning of any OpenBR program.

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
argc | int | Number of command line arguments.
argv[] | char * | Array of command line arguments.
sdk_path | const char * | (Optional) Path to the OpenBR sdk. If no path is provided OpenBR will try and find the sdk automatically.
use_gui | bool | Enable OpenBR to use make GUI windows. Default is false.

* **example call:** ```void br_initialize(int &argc, char *argv[], const char *sdk_path = "", bool use_gui = false)```
* **see:** [br_finalize](#br_finalize)

---

## br_initialize_default

Initializes the [Context](cpp_api.md#context) with default arguments.

* **return type:** void
* **parameters:** None
* **example call:** ```void br_initialize_default()```
* **see:** [br_finalize](#br_finalize)

---

## br_finalize

Finalizes the context. Required at the end of any OpenBR program.

* **return type:** void
* **parameters:** None
* **example call:** ```void br_finalize()```
* **see:** [br_initialize](#br_initialize)

---

## br_is_classifier

Checks if the provided algorithm is a classifier. Wrapper of [cpp_api.md#bool-isclassifierconst-qstring-algorithm].

* **return type:** bool
* **parameters:**

Parameter | Type | Description
--- | --- | ---
algorithm | const char * | Algorithm to check.

* **example call:** ```bool br_is_classifier(const char *algorithm)```

---

## br_make_mask

Constructs a [mask](../technical.md#the-evaluation-harness) from target and query inputs.

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
target_input | const char * | The target [Gallery](cpp_api.md#gallery)
query_input | const char * | The query [Gallery](cpp_api.md#gallery)
mask | const char * | The file to contain the resulting [mask](../technical.md#the-evaluation-harness).

* **example call:** ```void br_make_mask(const char *target_input, const char *query_input, const char *mask)```
* **see:** [br_combine_masks](#br_combine_masks)

---

## br_make_pairwise_mask

Constructs a [mask](../technical.md#the-evaluation-harness) from target and query inputs considering the target and input sets to be definite pairwise comparisons.

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
target_input | const char * | The target [Gallery](cpp_api.md#gallery)
query_input | const char * | The query [Gallery](cpp_api.md#gallery)
mask | const char * | The file to contain the resulting [mask](../technical.md#the-evaluation-harness).

* **example call:** ```void br_make_pairwise_mask(const char *target_input, const char *query_input, const char *mask)```
* **see:** [br_combine_masks](#br_combine_masks)

---

## br_most_recent_message

Returns the most recent line sent to stderr. Please see the bullet about input string buffers at the top of this page.

* **return type:** int
* **parameters:**

Parameter | Type | Description
--- | --- | ---
buffer | char * | Buffer to store the last line in.
buffer_length | int | Length of the buffer.

* **example call:** ```int br_most_recent_message(char * buffer, int buffer_length)```
* **see:** [br_progress](#br_progress), [br_time_remaining](#br_time_remaining)

---

## br_objects

Returns names and parameters for the requested objects. Each object is newline seperated. Arguments are seperated from the object name with a tab. This function uses [QRegExp](http://doc.qt.io/qt-5/QRegExp.html) syntax.

* **return type:** int
* **parameters:**

Parameter | Type | Description
--- | --- | ---
 buffer | char * | Output buffer for results.
 buffer_length | int | Length of output buffer.
 abstractions | const char * | (Optional) Regular expression of the abstractions to search. Default is ".*".
 implementations | const char * | (Optional) Regular expression of the implementations to search. Default is ".*".
 parameters | bool | (Optional) Include parameters after object name. Default is true.

* **example call:** ```int br_objects(char * buffer, int buffer_length, const char *abstractions = ".*", const char *implementations = ".*", bool parameters = true)```

---

## br_plot

Renders recognition performance figures for a set of **.csv** files created by [br_eval](#br_eval).

In order of their output, the figures are:
1. Metadata table
2. Receiver Operating Characteristic (ROC)
3. Detection Error Tradeoff (DET)
4. Score Distribution (SD) histogram
5. True Accept Rate Bar Chart (BC)
6. Cumulative Match Characteristic (CMC)
7. Error Rate (ERR) curve

Two files will be created:
 * **destination.R** which is the auto-generated R script used to render the figures.
 * **destination.pdf** which has all of the figures in one file multi-page file.

OpenBR uses file and folder names to automatically determine the plot legend.
For example, let's consider the case where three algorithms (<tt>A</tt>, <tt>B</tt>, & <tt>C</tt>) were each evaluated on two datasets (<tt>Y</tt> & <tt>Z</tt>).
The suggested way to plot these experiments on the same graph is to create a folder named <tt>Algorithm_Dataset</tt> that contains the six <tt>.csv</tt> files produced by br_eval <tt>A_Y.csv</tt>, <tt>A_Z.csv</tt>, <tt>B_Y.csv</tt>, <tt>B_Z.csv</tt>, <tt>C_Y.csv</tt>, & <tt>C_Z.csv</tt>.
The '<tt>_</tt>' character plays a special role in determining the legend title(s) and value(s).
In this case, <tt>A</tt>, <tt>B</tt>, & <tt>C</tt> will be identified as different values of type <tt>Algorithm</tt>, and each will be assigned its own color; <tt>Y</tt> & <tt>Z</tt> will be identified as different values of type Dataset, and each will be assigned its own line style.
Matches around the EER will be displayed if the matches parameter is set in [br_eval](#br_eval).

Returns **true** on success. Returns **false** on a failure to compile the figures due to a missing, out of date, or incomplete <tt>R</tt> installation.

This function requires a current [R](http://www.r-project.org/) installation with the following packages:

    install.packages(c("ggplot2", "gplots", "reshape", "scales", "jpg", "png"))

* **return type:** bool
* **parameters:**

Parameter | Type | Description
--- | --- | ---
num_files | int | Number of **.csv** files.
files[] | const char * | **.csv** files created using [br_eval](#br_eval).
destination | const char * | Basename for the resulting figures.
show | bool | Open **destination.pdf** using the system's default PDF viewer. Default is false.

* **example call:** ```bool br_plot(int num_files, const char *files[], const char *destination, bool show = false)```
* **see:** [br_eval](#br_eval)

---

## br_plot_detection

Renders detection performance figures for a set of **.csv** files created by [br_eval_detection](#br_eval_detection).

In order of their output, the figures are:
1. Discrete Receiver Operating Characteristic (DiscreteROC)
2. Continuous Receiver Operating Characteristic (ContinuousROC)
3. Discrete Precision Recall (DiscretePR)
4. Continuous Precision Recall (ContinuousPR)
5. Bounding Box Overlap Histogram (Overlap)
6. Average Overlap Table (AverageOverlap)
7. Average Overlap Heatmap (AverageOverlap)

Detection accuracy is measured with *overlap fraction = bounding box intersection / union*.
When computing *discrete* curves, an overlap >= 0.5 is considered a true positive, otherwise it is considered a false negative.
When computing *continuous* curves, true positives and false negatives are measured fractionally as *overlap* and *1-overlap* respectively.

Returns **true** on success. Returns **false** on a failure to compile the figures due to a missing, out of date, or incomplete <tt>R</tt> installation.

This function requires a current [R](http://www.r-project.org/) installation with the following packages:

    install.packages(c("ggplot2", "gplots", "reshape", "scales", "jpg", "png"))

* **return type:** bool
* **parameters:**

Parameter | Type | Description
--- | --- | ---
num_files | int | Number of **.csv** files.
files[] | const char * | **.csv** files created using [br_eval_detection](#br_eval_detection).
destination | const char * | Basename for the resulting figures.
show | bool | Open **destination.pdf** using the system's default PDF viewer. Default is false.

* **example call:** ```bool br_plot_detection(int num_files, const char *files[], const char *destination, bool show = false)```
* **see:** [br_eval_detection](#br_eval_detection), [br_plot](#br_plot)

---

## br_plot_landmarking

Renders landmarking performance figures for a set of **.csv** files created by [br_eval_landmarking](#br_eval_landmarking).

In order of their output, the figures are:
1. Cumulative landmarks less than normalized error (CD)
2. Normalized error box and whisker plots (Box)
3. Normalized error violin plots (Violin)

Landmarking error is normalized against the distance between two predifined points, usually inter-ocular distance (IOD).

* **return type:** bool
* **parameters:**

Parameter | Type | Description
--- | --- | ---
num_files | int | Number of **.csv** files.
files[] | const char * | **.csv** files created using [br_eval_landmarking](#br_eval_landmarking).
destination | const char * | Basename for the resulting figures.
show | bool | Open **destination.pdf** using the system's default PDF viewer. Default is false.

* **example call:** ```bool br_plot_landmarking(int num_files, const char *files[], const char *destination, bool show = false)```
* **see:** [br_eval_landmarking](#br_eval_landmarking), [br_plot](#br_plot)

---

## br_plot_metadata

Renders metadata figures for a set of **.csv** files with specified columns.

* **return type:** bool
* **parameters:**

Parameter | Type | Description
--- | --- | ---
num_files | int | Number of **.csv** files.
files[] | const char * | **.csv** files created by enrolling templates to **.csv** metadata files.
columns | const char * | ';' seperated list of columns to plot.
show | bool | Open **PlotMetadata.pdf** using the system's default PDF viewer.

* **example call:** ```bool br_plot_metadata(int num_files, const char *files[], const char *columns, bool show = false)```
* **see:** [br_plot](#br_plot)

---

## br_progress

Returns current progress from [Context](cpp_api.md#context)::progress().

* **return type:** float
* **parameters:** None
* **example call:** ```float br_progress()```
* **see:** [br_most_recent_message](#br_most_recent_message), [br_time_remaining](#br_time_remaining)

---

## br_read_pipe

Read and parse arguments from a named pipe. Used by the [command line api](cl_api.md) to implement **-daemon**, generally not useful otherwise. Guaranteed to return at least one argument. See the bullets at the top of this page on managed return values.

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
pipe | const char * | Pipe name
argc | int * | Argument count
argv | char *** | Argument list

* **example call:** ```void br_read_pipe(const char *pipe, int *argc, char ***argv)```

---

## br_scratch_path

Fills the buffer with the value of [Context](cpp_api.md#context)::scratchPath(). See the bullets at the top of this page on input string buffers.

* **return type:** int
* **parameters:**

Parameter | Type | Description
--- | --- | ---
buffer | char * | Buffer for scratch path
buffer_length | int | Length of buffer.

* **example call:** ```int br_scratch_path(char * buffer, int buffer_length)```
* **see:** [br_version](#br_version)

---

## br_sdk_path

Returns the full path to the root of the SDK.

* **return type:** const char *
* **parameters:** None
* **example call:** ```const char *br_sdk_path()```
* **see:** [br_initialize](#br_initialize)

---

## br_get_header

Retrieve the target and query inputs in the [BEE matrix](../technical.md#the-evaluation-harness) header. See the bullets at the top of this page on managed return values.

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
matrix | const char * | The [BEE matrix](../technical.md#the-evaluation-harness) file to modify
target_gallery | const char ** | The matrix target
query_gallery | const char ** | The matrix query

* **example call:** ```void br_get_header(const char *matrix, const char **target_gallery, const char **query_gallery)```
* **set:** [br_set_header](#br_set_header)

---

## br_set_header

Update the target and query inputs in the [BEE matrix](../technical.md#the-evaluation-harness) header.

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
matrix | const char * | The [BEE matrix](../technical.md#the-evaluation-harness) file to modify
target_gallery | const char ** | The matrix target
query_gallery | const char ** | The matrix query

* **example call:** ```void br_set_header(const char *matrix, const char *target_gallery, const char *query_gallery)```
* **see:** [br_get_header](#br_get_header)

---

## br_set_property

Appends the given value to the global [metadata](cpp_api.md#context) using the given key.

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
key | const char * | Key to append
value | const char * | Value to append

* **example call:** ```void br_set_property(const char *key, const char *value)```

---

## br_time_remaining

Returns estimate of time remaining in the current process.

* **return type:** int
* **parameters:** None
* **example call:** ```int br_time_remaining()```
* **see:** [br_most_recent_message](#br_most_recent_message), [br_progress](#br_progress)

---

## br_train

Trains the [Transform](cpp_api.md#transform) and [Distance](cpp_api.md#distance) on the input.

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
input | const char * | The [Gallery](cpp_api.md#gallery) to train on.
model | const char * | (Optional) String specifying the binary file to serialize training results to. The trained algorithm can be recovered by using this file as the algorithm. By default the trained algorithm will not be serialized to disk.

* **example call:** ```void br_train(const char *input, const char *model = "")```
* **see:** [br_train_n](#br_train_n)

---

## br_train_n

Convenience function for training on multiple inputs

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
num_inputs | int | Size of **inputs**
inputs[] | const char * | An array of [galleries](cpp_api.md#gallery) to train on.
modell | const char *model = | (Optional) String specifying the binary file to serialize training results to. The trained algorithm can be recovered by using this file as the algorithm. By default the trained algorithm will not be serialized to disk.

* **example call:** ```void br_train_n(int num_inputs, const char *inputs[], const char *model = "")```
* **see:** [br_train](#br_train)

---

## br_version

Returns the current OpenBR version.

* **return type:** const char *
* **parameters:** None
* **example call:** ```const char *br_version()```
* **see:** [br_about](#br_about), [br_scratch_path](#br_scratch_path)

---

## br_load_img

Load an image from a string buffer. This is an easy way to pass an image in memory from another programming language to openbr.

* **return type:** br_template
* **parameters:**

Parameter | Type | Description
--- | --- | ---
data | const char * | The image buffer.
len | int | The length of the buffer.

* **example call:** ```br_template br_load_img(const char *data, int len)```
* **see:** [br_unload_img](#br_unload_img)

---

## br_unload_img

Unload an image to a string buffer. This is an easy way to pass an image from openbr to another programming language.

* **return type:** unsigned char*
* **parameters:**

Parameter | Type | Description
--- | --- | ---
tmpl | br_template | Pointer to a [Template](cpp_api.md#template)

* **example call:** ```unsigned char* br_unload_img(br_template tmpl)```
* **see:** [br_load_img](#br_load_img)

---

## br_template_list_from_buffer

Deserialize a br::TemplateList from a buffer. Can be the buffer for a .gal file, since they are just a TemplateList serialized to disk.

* **return type:** br_template_list
* **parameters:**

Parameter | Type | Description
--- | --- | ---
buf | const char * | The buffer.
len | int | The length of the buffer.

* **example call:** ```br_template_list br_template_list_from_buffer(const char *buf, int len)```

---

## br_free_template

Free a [Template's](cpp_api.md#template) memory.

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
tmpl | br_template | Pointer to the [Template](cpp_api.md#template) to free.

* **example call:** ```void br_free_template(br_template tmpl)```

---

## br_free_template_list

Free a [TemplateList's](cpp_api.md#templatelist) memory.

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
tl | br_template_list | Pointer to the [TemplateList](cpp_api.md#templatelist) to free.

* **example call:** ```void br_free_template_list(br_template_list tl)```

---

## br_free_output

Free a [Output's](cpp_api.md#output) memory.

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
output | br_matrix_output | Pointer to the[Output](cpp_api.md#output) to free.

* **example call:** ```void br_free_output(br_matrix_output output)```

---

## br_img_rows

Returns the number of rows in an image.

* **return type:** int
* **parameters:**

Parameter | Type | Description
--- | --- | ---
tmpl | br_template | Pointer to a [Template](cpp_api.md#template).

* **example call:** ```int br_img_rows(br_template tmpl)```

---

## br_img_cols

Returns the number of cols in an image.

* **return type:** int
* **parameters:**

Parameter | Type | Description
--- | --- | ---
tmpl | br_template | Pointer to a [Template](cpp_api.md#template).

* **example call:** ```int br_img_cols(br_template tmpl)```

---

## br_img_channels

Returns the number of channels in an image.

* **return type:** int
* **parameters:**

Parameter | Type | Description
--- | --- | ---
tmpl | br_template | Pointer to a [Template](cpp_api.md#template).

* **example call:** ```int br_img_channels(br_template tmpl)```

---

## br_img_is_empty

Checks if the image is empty.

* **return type:** bool
* **parameters:**

Parameter | Type | Description
--- | --- | ---
tmpl | br_template | Pointer to a [Template](cpp_api.md#template).

* **example call:** ```bool br_img_is_empty(br_template tmpl)```

---

## br_get_filename

Get the filename for a [Template](cpp_api.md#template). Please see the bullets at the top of the page on input string buffers.

* **return type:** int
* **parameters:**

Parameter | Type | Description
--- | --- | ---
buffer | char * | Buffer to hold the filename
buffer_length | int | Length of the buffer
tmpl | br_template | Pointer to a [Template](cpp_api.md#template).

* **example call:** ```int br_get_filename(char * buffer, int buffer_length, br_template tmpl)```

---

## br_set_filename

Set the filename for a [Template](cpp_api.md#template).

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
tmpl | br_template | Pointer to a [Template](cpp_api.md#template).
filename | const char * | New filename for the template.

* **example call:** ```void br_set_filename(br_template tmpl, const char *filename)```

---

## br_get_metadata_string

Get metadata as a string for the given key in the given [Template](cpp_api.md#template).

* **return type:** int
* **parameters:**

Parameter | Type | Description
--- | --- | ---
buffer | char * | Buffer to hold the metadata string.
buffer_length | int | length of the buffer.
tmpl | br_template | Pointer to a [Template](cpp_api.md#template).
key | const char * | Key for the metadata lookup

* **example call:** ```int br_get_metadata_string(char * buffer, int buffer_length, br_template tmpl, const char *key)```

---

## br_enroll_template

Enroll a [Template](cpp_api.md#template) from the C API! Returns a pointer to a [TemplateList](cpp_api.md#templatelist)

* **return type:** br_template_list
* **parameters:**

Parameter | Type | Description
--- | --- | ---
tmpl | br_template | Pointer to a [Template](cpp_api.md#template).

* **example call:** ```br_template_list br_enroll_template(br_template tmpl)```

---

## br_enroll_template_list

Enroll a [TemplateList](cpp_api.md#templatelist) from the C API!

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
tl | br_template_list | Pointer to a [TemplateList](cpp_api.md#templatelist)

* **example call:** ```void br_enroll_template_list(br_template_list tl)```

---

## br_compare_template_lists

Compare [TemplateLists](cpp_api.md#templatelist) from the C API!

* **return type:** br_matrix_output
* **parameters:**

Parameter | Type | Description
--- | --- | ---
target | br_template_list | Pointer to a [TemplateList](cpp_api.md#templatelist)
query | br_template_list | Pointer to a [TemplateList](cpp_api.md#templatelist)

* **example call:** ```br_matrix_output br_compare_template_lists(br_template_list target, br_template_list query)```

---

## br_get_matrix_output_at

Get a value in the [MatrixOutput](cpp_api.md#matrixoutput).

* **return type:** float
* **parameters:**

Parameter | Type | Description
--- | --- | ---
output | br_matrix_output | Pointer to [MatrixOutput](cpp_api.md#matrixoutput)
row | int | Row for lookup
col | int | Col for lookup

* **example call:** ```float br_get_matrix_output_at(br_matrix_output output, int row, int col)```

---

## br_get_template

Get a pointer to a [Template](cpp_api.md#template) at a specified index.

* **return type:** br_template
* **parameters:**

Parameter | Type | Description
--- | --- | ---
tl | br_template_list | Pointer to a [TemplateList](cpp_api.md#templatelist)
index | int | Index into the template list. Should be in the range [0,len(tl) - 1].

* **example call:** ```br_template br_get_template(br_template_list tl, int index)```

---

## br_num_templates

Get the number of [Templates](cpp_api.md#template) in a [TemplateList](cpp_api.md#templatelist).

* **return type:** int
* **parameters:**

Parameter | Type | Description
--- | --- | ---
tl | br_template_list | Pointer to a [TemplateList](cpp_api.md#templatelist)

* **example call:** ```int br_num_templates(br_template_list tl)```

---

## br_make_gallery

Initialize a [Gallery](cpp_api.md#gallery).

* **return type:** br_gallery
* **parameters:**

Parameter | Type | Description
--- | --- | ---
gallery | const char * | String location of gallery on disk.

* **example call:** ```br_gallery br_make_gallery(const char *gallery)```

---

## br_load_from_gallery

Read [TemplateList](cpp_api.md#templatelist) from [Gallery](cpp_api.md#gallery).

* **return type:** br_template_list
* **parameters:**

Parameter | Type | Description
--- | --- | ---
gallery | br_gallery | Pointer to a [Gallery](cpp_api.md#gallery)

* **example call:** ```br_template_list br_load_from_gallery(br_gallery gallery)```

---

## br_add_template_to_gallery

Write a [Template](cpp_api.md#template) to the [Gallery](cpp_api.md#gallery) on disk

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
gallery | br_gallery | Pointer to a [Gallery](cpp_api.md#gallery)
tmpl | br_template | Pointer to a [Template](cpp_api.md#template)

* **example call:** ```void br_add_template_to_gallery(br_gallery gallery, br_template tmpl)```

---

## br_add_template_list_to_gallery

Write a [TemplateList](cpp_api.md#templatelist) to the [Gallery](cpp_api.md#gallery) on disk.

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
gallery | br_gallery | Pointer to a [Gallery](cpp_api.md#gallery)
tl | br_template_list | Pointer to a [TemplateList](cpp_api.md#templatelist)

* **example call:** ```void br_add_template_list_to_gallery(br_gallery gallery, br_template_list tl)```

---

## br_close_gallery

Close the [Gallery](cpp_api.md#gallery).

* **return type:** void
* **parameters:**

Parameter | Type | Description
--- | --- | ---
gallery | br_gallery | Pointer to a [Gallery](cpp_api.md#gallery)

* **example call:** ```void br_close_gallery(br_gallery gallery)```

---
