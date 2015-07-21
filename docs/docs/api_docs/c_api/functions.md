## br_about

Calls [Context](../cpp_api/context/context.md)::[about](../cpp_api/context/statics.md#about).

* **function definition:**

        const char *br_about()

* **parameters:** None
* **output:** (const char *) Returns a string describing OpenBR
* **see:** [br_version](#br_version)

---

## br_cat

Concatenates a list of galleries into 1 gallery.

* **function definition:**

        void br_cat(int num_input_galleries, const char *input_galleries[], const char *output_gallery)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    num_input_galleries | int | Size of input_galleries
    input_galleries[] | const char * | List of galleries
    output_gallery | const char * | Pointer to store concatenated gallery

* **output:** void
* **see:** [Cat](../cpp_api/apifunctions.md#cat)

---

## br_deduplicate

Removes duplicate [templates](../cpp_api/template/template.md) in a [gallery](../cpp_api/gallery/gallery.md). If a galley contains n duplicates, the first n-1 duplicates in the gallery will be removed and the nth will be kept. Users are encouraged to use binary gallery formats as the entire gallery is read into memory in one call to [Gallery](../cpp_api/gallery/gallery.md)::[read](../cpp_api/gallery/functions.md#read).

* **function definition:**

        void br_deduplicate(const char *input_gallery, const char *output_gallery, const char *threshold)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    input_gallery | const char * | Gallery to be deduplicated
    output_gallery | const char * | Deduplicated gallery
    threshold | const char * | Comparisons with a match score >= this value are designated to be duplicates.

* **output:** (void)

---

## br_cluster

Clusters one or more similarity matrices into a list of subjects. A [similarity matrix](../../tutorials.md#the-evaluation-harness) is a type of [Output](../cpp_api/output/output.md). The current clustering algorithm is a simplified implementation of the algorithm proposed by Zhu et al[^1].

* **function definition:**

        void br_cluster(int num_simmats, const char *simmats[], float aggressiveness, const char *csv)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    num_simmats | int | Size of **simmats**
    simmats[] | const char * | Array of [simmat](../../tutorials.md#the-evaluation-harness) composing one large self-similarity matrix arranged in row major order.
    aggressiveness | float | The higher the aggressiveness the larger the clusters. Suggested range is [0,10]
    csv | const char * | The cluster results file to generate. Results are stored one row per cluster and use gallery indices.

* **output:** (void)

---

## br_combine_masks

Combines several equal-sized mask matrices. A comparison may not be simultaneously indentified as both a genuine and an imposter by different input masks.

* **function definition:**

    void br_combine_masks(int num_input_masks, const char *input_masks[], const char *output_mask, const char *method)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    num_input_masks | int | Size of **input_masks**
    input_masks[] | const char * | Array of [mask matrices](../../tutorials.md#the-evaluation-harness) to combine. All matrices must have the same dimensions.
    output_mask | const char * | The file to contain the resulting [mask matrix](../../tutorials.md#the-evaluation-harness)
    method | const char * | Possible values are: <ul><li>And - Ignore comparison if *any* input masks ignore.</li> <li>Or - Ignore comparison if *all* input masks ignore.</li></ul>

* **see:** [br_make_mask](#br_make_mask)

---

## br_compare

Compares each [Template](../cpp_api/template/template.md) in the query [Gallery](../cpp_api/gallery/gallery.md) to each [Template](../cpp_api/template/template.md)  in the target [Gallery](../cpp_api/gallery/gallery.md).

* **function definition:**

        void br_compare(const char *target_gallery, const char *query_gallery, const char *output = "")

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    target_gallery | const char * | target_gallery The [Gallery](../cpp_api/gallery/gallery.md) file whose templates make up the columns of the output.
    query_gallery | const char * | The [Gallery](../cpp_api/gallery/gallery.md) file whose templates make up the rows of the output. A value of '.' reuses the target gallery as the query gallery.
    output | const char * | (Optional) The [Output](../cpp_api/output/output.md) file to contain the results of comparing the templates. The default behavior is to print scores to the terminal.

* **output:** (void)
* **see:** br_enroll

---

## br_compare_n

Convenience function for comparing to multiple targets.

* **function definition:**

        void br_compare_n(int num_targets, const char *target_galleries[], const char *query_gallery, const char *output)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    num_targets | int | Size of **target_galleries**
    target_galleries[] | const char * | Target galleries to compare against
    query_gallery | const char * | query gallery for comparison.
    output | const char * | (Optional) [Output](../cpp_api/output/output.md) file to contain the results of comparing the templates. The default behavior is to print scores to the terminal.

* **output:** (void)
* **see:** br_compare

---

## br_pairwise_compare

DOCUMENT ME!

* **function definition:**

        void br_pairwise_compare(const char *target_gallery, const char *query_gallery, const char *output = "")

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    target_gallery | const char * | DOCUMENT ME
    query_gallery | const char * | DOCUMENT ME
    output | const char * | DOCUMENT ME

* **output:** (void)

---

## br_convert

Convert a file to a different type. Files can only be converted to types within the same group. For example [formats](../cpp_api/format/format.md) can only be converted to other [formats](../cpp_api/format/format.md).

* **function definition:**

        void br_convert(const char *file_type, const char *input_file, const char *output_file)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    file_type | const char * | Type of file to convert. Options are [Format](../cpp_api/format/format.md), [Gallery](../cpp_api/gallery/gallery.md) or [Output](../cpp_api/output/output.md).
    input_file | const char * | File to convert.
    output_file | const char * | Output file. Type is determined by the file extension.

* **output:** (void)

---

## br_enroll

Constructs [Template(s)](../cpp_api/template/template.md) from an input.

* **function definition:**

        void br_enroll(const char *input, const char *gallery = "")

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    input | const char * | The [format](../cpp_api/format/format.md) or [gallery](../cpp_api/gallery/gallery.md) to enroll.
    gallery | const char * | (Optional) The [Gallery](../cpp_api/gallery/gallery.md) file to contain the enrolled templates. By default the gallery will be held in memory and *input* can used as a gallery in [br_compare](#br_compare)

* **output:** (void)
* **see:** [br_enroll_n](#br_enroll_n)

---

## br_enroll_n

Convenience function for enrolling multiple inputs.

* **function definition:**

        void br_enroll_n(int num_inputs, const char *inputs[], const char *gallery = "")

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    num_inputs | int | Size of **inputs**.
    inputs[] | const char * | Array of inputs to enroll.
    gallery | const char * | (Optional) The [Gallery](../cpp_api/gallery/gallery.md) file to contain the enroll templates.

* **output:** (void)
* **see:** [br_enroll](#br_enroll)

---

## br_project

A naive alternative to [br_enroll](#br_enroll).

* **function definition:**

        void br_project(const char *input, const char *output)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    input | const char * | The [format](../cpp_api/format/format.md) or [gallery](../cpp_api/gallery/gallery.md) to enroll.
    output | const char * | The [Gallery](../cpp_api/gallery/gallery.md) file to contain the enrolled templates. By default the gallery will be held in memory and *input* can used as a gallery in [br_compare](#br_compare)

* **output:** (void)
* **see:** [br_enroll](#br_enroll)

---

## br_eval

Creates a **.csv** file containing performance metrics from evaluating the similarity matrix using the mask matrix.

* **function defintion:**

        float br_eval(const char *simmat, const char *mask, const char *csv = "", int matches = 0)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    simmat | const char * | The [simmat](../../tutorials.md#the-evaluation-harness) to use
    mask | const char * | The [mask](../../tutorials.md#the-evaluation-harness) to use.
    csv | const char * | (Optional) The **.csv** file to contain performance metrics.
    matches | int | (Optional) An integer number of matches to output around the EER. Default is 0.

* **output:** (float) Returns the true accept rate (TAR) at a false accept rate (FAR) of one in one thousand
* **see:** [br_plot](#br_plot)

---

## br_assert_eval

Evaluates the similarity matrix using the mask matrix.  Function aborts if TAR @ FAR = 0.001 does not meet an expected performance value.

* **function definition:**

        void br_assert_eval(const char *simmat, const char *mask, const float accuracy)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    simmat | const char * | The [simmat](../../tutorials.md#the-evaluation-harness) to use
    mask | const char * | The [mask](../../tutorials.md#the-evaluation-harness)
    accuracy | const float | Desired true accept rate at false accept rate of one in one thousand.

* **output:** (void)

---

## br_inplace_eval

Creates a **.csv** file containing performance metrics from evaluating the similarity matrix using galleries containing ground truth labels.

* **function definition:**

        float br_inplace_eval(const char * simmat, const char *target, const char *query, const char *csv = "")

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    simmat | const char * | The [simmat](../../tutorials.md#the-evaluation-harness)
    target | const char * | The name of a gallery containing metadata for the target set.
    query | const char * | The name of a gallery containing metadata for the query set.
    csv | const char * | (Optional) The **.csv** file to contain performance metrics.

* **output:** (float) Returns the true accept rate (TAR) at a false accept rate (FAR) of one in one thousand
* **see:** [br_plot](#br_plot)

---

## br_eval_classification

Evaluates and prints classification accuracy to terminal.

* **function definition:**

        void br_eval_classification(const char *predicted_gallery, const char *truth_gallery, const char *predicted_property = "", const char *truth_property = "")

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    predicted_gallery | const char * | The predicted [Gallery](../cpp_api/gallery/gallery.md).
    truth_gallery | const char * | The ground truth [Gallery](../cpp_api/gallery/gallery.md).
    predicted_property | const char * | (Optional) Which metadata key to use from the **predicted_gallery**.
    truth_property | const char * | (Optional) Which metadata key to use from the **truth_gallery**.

* **output:** (void)

---

## br_eval_clustering

Evaluates and prints clustering accuracy to the terminal.

* **function definition:**

        void br_eval_clustering(const char *csv, const char *gallery, const char * truth_property)

* **parameters:**

Parameter | Type | Description
--- | --- | ---
csv | const char * | The cluster results file.
gallery | const char * | The [Gallery](../cpp_api/gallery/gallery.md) used to generate the [simmat](../../tutorials.md#the-evaluation-harness) that was clustered.
truth_property | const char * | (Optional) which metadata key to use from **gallery**, defaults to Label

* **output:** (void)

---

## br_eval_detection

Evaluates and prints detection accuracy to terminal.

* **function definition:**

        float br_eval_detection(const char *predicted_gallery, const char *truth_gallery, const char *csv = "", bool normalize = false, int minSize = 0, int maxSize = 0)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    predicted_gallery | const char * | The predicted [Gallery](../cpp_api/gallery/gallery.md).
    truth_gallery | const char * | The ground truth [Gallery](../cpp_api/gallery/gallery.md).
    csv | const char * | (Optional) The **.csv** file to contain performance metrics.
    normalize | bool | (Optional) Flag to normalize predicted bounding boxes for improved detection. Defaults to false.
    minSize | int | (Optional) Minimum size of faces to be considered in the evaluation. Size is applied to predicted and ground truth galleries. Defaults to -1 (no minimum size).
    maxSize | int | (Optional) Maximum size if faces to be considered in the evaluation. Size is applied to predicted and ground truth galleries. Defaults to -1 (no maximum size).

* **output:** (float) Returns the true accept rate (TAR) at a false accept rate (FAR) of one in one thousand

---

## br_eval_landmarking

Evaluates and prints landmarking accuracy to terminal.

* **function definition:**

        float br_eval_landmarking(const char *predicted_gallery, const char *truth_gallery, const char *csv = "", int normalization_index_a = 0, int normalization_index_b = 1, int sample_index = 0, int total_examples = 5)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    predicted_gallery | const char * | The predicted [Gallery](../cpp_api/gallery/gallery.md).
    truth_gallery | const char * | The ground truth [Gallery](../cpp_api/gallery/gallery.md).
    csv | const char * | (Optional) The **.csv** file to contain performance metrics.
    normalization_index_a | int | (Optional) The first index in the list of points to use for normalization. Default is 0.
    normalization_index_b | int | (Optional) The second index in the list of points to use for normalization. Default is 1.
    sample_index | int | (Optional) The index for sample landmark image in ground truth gallery. Default = 0.
    total_examples | int | (Optional) The number of accurate and inaccurate examples to display. Default is 5.

* **output:** (float) Returns the true accept rate (TAR) at a false accept rate (FAR) of one in one thousand

---

## br_eval_regression

Evaluates regression accuracy to disk.

* **function definition:**

        void br_eval_regression(const char *predicted_gallery, const char *truth_gallery, const char *predicted_property = "", const char *truth_property = "")

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    predicted_gallery | const char * | The predicted [Gallery](../cpp_api/gallery/gallery.md)
    truth_gallery | const char * | The ground truth [Gallery](../cpp_api/gallery/gallery.md)
    predicted_property | const char * | (Optional) Which metadata key to use from **predicted_gallery**.
    truth_property | const char * | (Optional) Which metadata key to use from **truth_gallery**.

* **output:** (void)

---

## br_fuse

Perform score level fusion on similarity matrices.

* **function definition:**

        void br_fuse(int num_input_simmats, const char *input_simmats[], const char *normalization, const char *fusion, const char *output_simmat)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    num_input_simmats | int | Size of **input_simmats**.
    input_simmats[] | const char * | Array of [simmats](../../tutorials.md#the-evaluation-harness). All simmats must have the same dimensions.
    normalization | const char * | Valid options are: <ul> <li>None - No score normalization.</li> <li>MinMax - Scores normalized to [0,1].</li> <li>ZScore - Scores normalized to a standard normal curve.</li> </ul>
    fusion | const char * | Valid options are: <ul> <li>Min - Uses the minimum score.</li> <li>Max - Uses the maximum score.</li> <li>Sum - Sums the scores. Sums can also be weighted: <tt>SumW1:W2:...:Wn</tt>.</li> <li>Replace - Replaces scores in the first matrix with scores in the second matrix when the mask is set.</li> </ul>
    output_simmat | const char * | [Simmat](../../tutorials.md#the-evaluation-harness) to contain the fused scores.

* **output:** (void)

---

## br_initialize

Initializes the [Context](../cpp_api/context/context.md). Required at the beginning of any OpenBR program.

* **function definition:**

        void br_initialize(int &argc, char *argv[], const char *sdk_path = "", bool use_gui = false)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    argc | int | Number of command line arguments.
    argv[] | char * | Array of command line arguments.
    sdk_path | const char * | (Optional) Path to the OpenBR sdk. If no path is provided OpenBR will try and find the sdk automatically.
    use_gui | bool | (Optional) Enable OpenBR to use make GUI windows. Default is false.

* **output:** (void)
* **see:** [br_finalize](#br_finalize)

---

## br_initialize_default

Initializes the [Context](../cpp_api/context/context.md) with default arguments.

* **function definition:**

        void br_initialize_default()

* **parameters:** None
* **output:** (void)
* **see:** [br_finalize](#br_finalize)

---

## br_finalize

Finalizes the context. Required at the end of any OpenBR program.

* **function definition:**

        void br_finalize()

* **parameters:** None
* **output:** (void)
* **see:** [br_initialize](#br_initialize)

---

## br_is_classifier

Checks if the provided algorithm is a classifier. Wrapper of [IsClassifier](../cpp_api/apifunctions.md#isclassifier).

* **function definition:**

        bool br_is_classifier(const char *algorithm)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    algorithm | const char * | Algorithm to check.

* **output:** (bool) Returns true if the algorithm is a classifier (does not have an associated distance)
* **see:** [IsClassifier](../cpp_api/apifunctions.md#isclassifier)

---

## br_make_mask

Constructs a [mask](../../tutorials.md#the-evaluation-harness) from target and query inputs.

* **function definition:**

        void br_make_mask(const char *target_input, const char *query_input, const char *mask)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    target_input | const char * | The target [Gallery](../cpp_api/gallery/gallery.md)
    query_input | const char * | The query [Gallery](../cpp_api/gallery/gallery.md)
    mask | const char * | The file to contain the resulting [mask](../../tutorials.md#the-evaluation-harness).

* **output:** (void)
* **see:** [br_combine_masks](#br_combine_masks)

---

## br_make_pairwise_mask

Constructs a [mask](../../tutorials.md#the-evaluation-harness) from target and query inputs considering the target and input sets to be definite pairwise comparisons.

* **function definition:**

        void br_make_pairwise_mask(const char *target_input, const char *query_input, const char *mask)

* **parameters:**

Parameter | Type | Description
--- | --- | ---
target_input | const char * | The target [Gallery](../cpp_api/gallery/gallery.md)
query_input | const char * | The query [Gallery](../cpp_api/gallery/gallery.md)
mask | const char * | The file to contain the resulting [mask](../../tutorials.md#the-evaluation-harness).

* **output:** (void)
* **see:** [br_combine_masks](#br_combine_masks)

---

## br_most_recent_message

Returns the most recent line sent to stderr. For information on input string buffers please look [here](../c_api.md#input-string-buffers)

* **function definition:**

        int br_most_recent_message(char * buffer, int buffer_length)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    buffer | char * | Buffer to store the last line in.
    buffer_length | int | Length of the buffer.

* **output:** (int) Returns the required size of the input buffer for the most recent message to fit completely
* **see:** [br_progress](#br_progress), [br_time_remaining](#br_time_remaining)

---

## br_objects

Returns names and parameters for the requested objects. Each object is newline seperated. Arguments are seperated from the object name with a tab. This function uses [QRegExp][QRegExp] syntax.

* **function definition:**

        int br_objects(char * buffer, int buffer_length, const char *abstractions = ".*", const char *implementations = ".*", bool parameters = true)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    buffer | char * | Output buffer for results.
    buffer_length | int | Length of output buffer.
    abstractions | const char * | (Optional) Regular expression of the abstractions to search. Default is ".\*".
    implementations | const char * | (Optional) Regular expression of the implementations to search. Default is ".\*".
    parameters | bool | (Optional) Include parameters after object name. Default is true.

* **output:** (int) Returns the required size of the input buffer for the returned objects to fit completely

---

## br_plot

Renders recognition performance figures for a set of **.csv** files created by [br_eval](#br_eval).

In order of their output, the figures are:
1. Metadata table
2. Receiver Operating Characteristic (ROC)
3. Detection Error Tradeoff (DET)
4. Identification Error Tradeoff (IET)
5. Cumulative Match Characteristic (CMC)
6. Score Distribution (SD) histogram
7. True Accept Rate Bar Chart (BC)
8. Error Rate (ERR) curve

Two files will be created:
 * **destination.R** which is the auto-generated R script used to render the figures.
 * **destination.pdf** which has all of the figures in one file multi-page file (note that **destination%1d.png** will output each figure to a separate file).

OpenBR uses file and folder names to automatically determine the plot legend.
For example, let's consider the case where three algorithms (<tt>A</tt>, <tt>B</tt>, & <tt>C</tt>) were each evaluated on two datasets (<tt>Y</tt> & <tt>Z</tt>).
The suggested way to plot these experiments on the same graph is to create a folder named <tt>Algorithm_Dataset</tt> that contains the six <tt>.csv</tt> files produced by br_eval <tt>A_Y.csv</tt>, <tt>A_Z.csv</tt>, <tt>B_Y.csv</tt>, <tt>B_Z.csv</tt>, <tt>C_Y.csv</tt>, & <tt>C_Z.csv</tt>.
The '<tt>_</tt>' character plays a special role in determining the legend title(s) and value(s).
In this case, <tt>A</tt>, <tt>B</tt>, & <tt>C</tt> will be identified as different values of type <tt>Algorithm</tt>, and each will be assigned its own color; <tt>Y</tt> & <tt>Z</tt> will be identified as different values of type Dataset, and each will be assigned its own line style.
Matches around the EER will be displayed if the matches parameter is set in [br_eval](#br_eval).

It is possible to customize some aspects of your plots using the [File](../cpp_api/file/file.md) key/value metadata convention; possible keys are described below.


Key             | Value          | Description
---             | ----           | -----------
smooth          | [QString]      | The file pivot to average across evaluation splits.  Typically "Dataset" if using the folder name from above.
ncol            | int            | Number of columns in plot legends.
confidence      | float          | Confidence interval calculated for smooth, defaults to `0.95`
metadata        | bool           | Optional plot metadata, defaults to `true`
csv             | bool           | Optional output metadata tables to csv, defaults to `false`
\*Options       | [QStringList]  | Key/value list of options for a specific plot.  Plots include "roc", "det", "iet", "cmc"

Specific plot options are described below:

Key             | Value          | Description
---             | ----           | -----------
title           | [QString]      | Plot title
size            | float          | Line width
legendPosition  | [QPointF]      | Legend coordinates on plot, ex. legendPosition=(X,Y)
textSize        | float          | Size of text for title, legend and axes
xTitle/yTitle   | [QString]      | Title for x/y axis
xLog/yLog       | bool           | Plot log scale for x/y axis
xLimits/yLimits | [QPointF]      | Set x/y axis limits, ex. xLimits=(lower,upper)
xLabels/yLabels | [QString]      | Labels for ticks on x/y axis, ex. xLabeles=percent or xLabels=(1,5,10)
xBreaks/yBreaks | [QString]      | Specify breaks/ticks on x/y axis, ex. xBreaks=pretty_breaks(n=10) or xBreaks=(1,5,10)

If specifying plot options it is a good idea to wrap the destination file in single quotes to avoid parsing errors.
The example below plots plots the six br_eval results in the Algorithm_Dataset folder described above, sets the number of legend columns and specifies some options for the CMC plot.

`br -plot Algorithm_Dataset/* 'destination.pdf[ncol=3,cmcOptions=[xLog=false,xLimits=(1,20),xBreaks=pretty_breaks(n=10),xTitle=Ranks 1 through 20]]'`

This function requires a current [R][R] installation with the following packages:

        install.packages(c("ggplot2", "gplots", "reshape", "scales", "jpg", "png"))

* **function definiton:**

        bool br_plot(int num_files, const char *files[], const char *destination, bool show = false)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    num_files | int | Number of **.csv** files.
    files[] | const char * | **.csv** files created using [br_eval](#br_eval).
    destination | const char * | Basename for the resulting figures.
    show | bool | Open **destination.pdf** using the system's default PDF viewer. Default is false.

* **output:** (bool) Returns true on success. Returns false on a failure to compile the figures due to a missing, out of date, or incomplete <tt>R</tt> installation.
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

This function requires a current [R](http://www.r-project.org/) installation with the following packages:

    install.packages(c("ggplot2", "gplots", "reshape", "scales", "jpg", "png"))

* **function definition:**

        bool br_plot_detection(int num_files, const char *files[], const char *destination, bool show = false)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    num_files | int | Number of **.csv** files.
    files[] | const char * | **.csv** files created using [br_eval_detection](#br_eval_detection).
    destination | const char * | Basename for the resulting figures.
    show | bool | Open **destination.pdf** using the system's default PDF viewer. Default is false.

* **output:** (bool) Returns true on success. Returns false on a failure to compile the figures due to a missing, out of date, or incomplete <tt>R</tt> installation.
* **see:** [br_eval_detection](#br_eval_detection), [br_plot](#br_plot)

---

## br_plot_landmarking

Renders landmarking performance figures for a set of **.csv** files created by [br_eval_landmarking](#br_eval_landmarking).

In order of their output, the figures are:
1. Cumulative landmarks less than normalized error (CD)
2. Normalized error box and whisker plots (Box)
3. Normalized error violin plots (Violin)

Landmarking error is normalized against the distance between two predifined points, usually inter-ocular distance (IOD).

* **function definition:**

        bool br_plot_landmarking(int num_files, const char *files[], const char *destination, bool show = false)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    num_files | int | Number of **.csv** files.
    files[] | const char * | **.csv** files created using [br_eval_landmarking](#br_eval_landmarking).
    destination | const char * | Basename for the resulting figures.
    show | bool | Open **destination.pdf** using the system's default PDF viewer. Default is false.

* **output:** (bool) Returns true on success. Returns false on a failure to compile the figures due to a missing, out of date, or incomplete <tt>R</tt> installation.
* **see:** [br_eval_landmarking](#br_eval_landmarking), [br_plot](#br_plot)

---

## br_plot_metadata

Renders metadata figures for a set of **.csv** files with specified columns.

* **function definition:**

        bool br_plot_metadata(int num_files, const char *files[], const char *columns, bool show = false)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    num_files | int | Number of **.csv** files.
    files[] | const char * | **.csv** files created by enrolling templates to **.csv** metadata files.
    columns | const char * | ';' seperated list of columns to plot.
    show | bool | Open **PlotMetadata.pdf** using the system's default PDF viewer.

* **output:** (bool) Returns true on success. Returns false on a failure to compile the figures due to a missing, out of date, or incomplete <tt>R</tt> installation.
* **see:** [br_plot](#br_plot)

---

## br_progress

Returns current progress from [Context](../cpp_api/context/context.md)::[progress](../cpp_api/context/functions.md#progress).

* **function definition:**

        float br_progress()

* **parameters:** None

* **output:** (float) Returns the completion percentage of the running process
* **see:** [br_most_recent_message](#br_most_recent_message), [br_time_remaining](#br_time_remaining)

---

## br_read_pipe

Read and parse arguments from a named pipe. Used by the [command line api](../cl_api.md) to implement **-daemon**, generally not useful otherwise. Guaranteed to return at least one argument. For information on managed returned values see [here](../c_api.md#memory)

* **function defintion:**

        void br_read_pipe(const char *pipe, int *argc, char ***argv)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    pipe | const char * | Pipe name
    argc | int * | Argument count
    argv | char *** | Argument list

* **output:** (void)

---

## br_scratch_path

Fills the buffer with the value of [Context](../cpp_api/context/context.md)::[scratchPath](../cpp_api/context/statics.md#scratchpath). For information on input string buffers see [here](../c_api.md#input-string-buffers).

* **function defintion:**

        int br_scratch_path(char * buffer, int buffer_length)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    buffer | char * | Buffer for scratch path
    buffer_length | int | Length of buffer.

* **output:** (int) Returns the required size of the input buffer for the most recent message to fit completely
* **see:** [br_version](#br_version)

---

## br_sdk_path

Returns the full path to the root of the SDK.

* **function definition:**

        const char *br_sdk_path()

* **parameters:** None
* **output:** (const char *) Returns the full path to the root of the SDK
* **see:** [br_initialize](#br_initialize)

---

## br_get_header

Retrieve the target and query inputs in the [BEE matrix](../../tutorials.md#the-evaluation-harness) header. For information on managed return values see [here](../c_api.md#memory).

* **function definition:**

        void br_get_header(const char *matrix, const char **target_gallery, const char **query_gallery)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    matrix | const char * | The [BEE matrix](../../tutorials.md#the-evaluation-harness) file to modify
    target_gallery | const char ** | The matrix target
    query_gallery | const char ** | The matrix query

* **output:** (void)
* **set:** [br_set_header](#br_set_header)

---

## br_set_header

Update the target and query inputs in the [BEE matrix](../../tutorials.md#the-evaluation-harness) header.

* **function definition:**

        void br_set_header(const char *matrix, const char *target_gallery, const char *query_gallery)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    matrix | const char * | The [BEE matrix](../../tutorials.md#the-evaluation-harness) file to modify
    target_gallery | const char ** | The matrix target
    query_gallery | const char ** | The matrix query

* **output:** (void)
* **see:** [br_get_header](#br_get_header)

---

## br_set_property

Appends a provided value to the [global metadata](../cpp_api/context/context.md) using a provided key.

* **function definition:**

        void br_set_property(const char *key, const char *value)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    key | const char * | Key to append
    value | const char * | Value to append

* **output:** (void)

---

## br_time_remaining

Returns estimate of time remaining in the current process.

* **function definition:**

        int br_time_remaining()

* **parameters:** None
* **output:** (int) Returns an estimate of the time remaining
* **see:** [br_most_recent_message](#br_most_recent_message), [br_progress](#br_progress)

---

## br_train

Trains a provided model's [Transform](../cpp_api/transform/transform.md) and [Distance](../cpp_api/distance/distance.md) on the provided input.

* **function definiton:**

        void br_train(const char *input, const char *model = "")

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    input | const char * | The [Gallery](../cpp_api/gallery/gallery.md) to train on.
    model | const char * | (Optional) String specifying the binary file to serialize training results to. The trained algorithm can be recovered by using this file as the algorithm. By default the trained algorithm will not be serialized to disk.

* **output:** (void)
* **see:** [br_train_n](#br_train_n)

---

## br_train_n

Convenience function for training on multiple inputs

* **function definition:**

        void br_train_n(int num_inputs, const char *inputs[], const char *model = "")

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    num_inputs | int | Size of **inputs**
    inputs[] | const char * | An array of [galleries](../cpp_api/gallery/gallery.md) to train on.
    model | const char * | (Optional) String specifying the binary file to serialize training results to. The trained algorithm can be recovered by using this file as the algorithm. By default the trained algorithm will not be serialized to disk.

* **output:** (void)
* **see:** [br_train](#br_train)

---

## br_version

Get the current OpenBR version.

* **function definition:**

        const char *br_version()

* **parameters:** None
* **output:** (const char *) Returns the current OpenBR version
* **see:** [br_about](#br_about), [br_scratch_path](#br_scratch_path)

---

## br_slave_process

For internal use via [ProcessWrapperTransform](../../plugin_docs/core.md#processwrappertransform)

* **function definition:**

        void br_slave_process(const char * baseKey)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    baseKey | const char * | base key for the slave process

* **output:** (void)

---

## br_load_img

Load an image from a string buffer. This is an easy way to pass an image in memory from another programming language to openbr.

* **function definition:**

        br_template br_load_img(const char *data, int len)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    data | const char * | The image buffer.
    len | int | The length of the buffer.

* **output:** ([br_template](typedefs.md#br_template) Returns a [br_template](typedefs.md#br_template) loaded with the provided image
* **see:** [br_unload_img](#br_unload_img)

---

## br_unload_img

Unload an image to a string buffer. This is an easy way to pass an image from openbr to another programming language.

* **function definition:**

        unsigned char* br_unload_img(br_template tmpl)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    tmpl | [br_template](typedefs.md#br_template) | Pointer to a [Template](../cpp_api/template/template.md)

* **output:** (unsigned char*)  Returns a buffer loaded with the image data from tmpl
* **see:** [br_load_img](#br_load_img)

---

## br_template_list_from_buffer

Deserialize a [TemplateList](../cpp_api/templatelist/templatelist.md) from a buffer. Can be the buffer for a .gal file, since they are just a [TemplateList](../cpp_api/templatelist/templatelist.md) serialized to disk.

* **function definition:**

        br_template_list br_template_list_from_buffer(const char *buf, int len)

* **return type:** br_template_list
* **parameters:**

Parameter | Type | Description
--- | --- | ---
buf | const char * | The buffer.
len | int | The length of the buffer.

* **output:** ([br_template_list](typedefs.md#br_template_list)) Returns a pointer to a [TemplateList](../cpp_api/templatelist/templatelist.md) created from the buffer.

---

## br_free_template

Free a [Template's](../cpp_api/template/template.md) memory.

* **function definition:**

        void br_free_template(br_template tmpl)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    tmpl | br_template | Pointer to the [Template](../cpp_api/template/template.md) to free.

* **output:** (void)

---

## br_free_template_list

Free a [TemplateList's](../cpp_api/templatelist/templatelist.md) memory.

* **function definition:**

        void br_free_template_list(br_template_list tl)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    tl | [br_template_list](typedefs.md#br_template_list) | Pointer to the [TemplateList](../cpp_api/templatelist/templatelist.md) to free.

* **output:** (void)

---

## br_free_output

Free a [Output's](../cpp_api/output/output.md) memory.

* **function definition:**

        void br_free_output(br_matrix_output output)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    output | [br_matrix_output](typedefs.md#br_matrix_output) | Pointer to the[Output](../cpp_api/output/output.md) to free.

* **output:** (void)

---

## br_img_rows

Returns the number of rows in an image.

* **function definition:**

        int br_img_rows(br_template tmpl)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    tmpl | [br_template](typedefs.md#br_template) | Pointer to a [Template](../cpp_api/template/template.md).

* **output:** (int) Returns the number of rows in an image

---

## br_img_cols

Returns the number of cols in an image.

* **function definition:**

        int br_img_cols(br_template tmpl)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    tmpl | [br_template](typedefs.md#br_template) | Pointer to a [Template](../cpp_api/template/template.md).

* **output:** (int) Returns the number of columns in an image

---

## br_img_channels

Returns the number of channels in an image.

* **function definition:**

        int br_img_channels(br_template tmpl)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    tmpl | [br_template](typedefs.md#br_template) | Pointer to a [Template](../cpp_api/template/template.md).

* **output:** (int) Returns the number of channels in an image

---

## br_img_is_empty

Checks if the image is empty.

* **function definition:**

        bool br_img_is_empty(br_template tmpl)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    tmpl | [br_template](typedefs.md#br_template) | Pointer to a [Template](../cpp_api/template/template.md).

* **output:** (bool) Returns true if the image is empty, false otherwise.

---

## br_get_filename

Get the name of the [file](../cpp_api/template/members.md#file) of a provided [Template](../cpp_api/template/template.md). For information on input string buffers please see [here](../c_api.md#input-string-buffers)

* **function definition:**

        int br_get_filename(char * buffer, int buffer_length, br_template tmpl)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    buffer | char * | Buffer to hold the filename
    buffer_length | int | Length of the buffer
    tmpl | [br_template](typedefs.md#br_template) | Pointer to a [Template](../cpp_api/template/template.md).

* **output:** (int) Returns the size of the buffer required to hold the entire file name.

---

## br_set_filename

Set the name of the [file](../cpp_api/template/members.md#file) for a provided [Template](../cpp_api/template/template.md).

* **function definition:**

        void br_set_filename(br_template tmpl, const char *filename)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    tmpl | [br_template](typedefs.md#br_template) | Pointer to a [Template](../cpp_api/template/template.md).
    filename | const char * | New filename for the template.

* **output:** (void)

---

## br_get_metadata_string

Get the [metadata](../cpp_api/file/members.md#m_metadata) value as a string for a provided key in a provided [Template](../cpp_api/template/template.md).

* **function definition:**

        int br_get_metadata_string(char * buffer, int buffer_length, br_template tmpl, const char *key)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    buffer | char * | Buffer to hold the metadata string.
    buffer_length | int | length of the buffer.
    tmpl | [br_template](typedefs.md#br_template) | Pointer to a [Template](../cpp_api/template/template.md).
    key | const char * | Key for the metadata lookup

* **output:** (int) Returns the size of the buffer required to hold the entire metadata string

---

## br_enroll_template

Enroll a [Template](../cpp_api/template/template.md) from the C API!

* **function definition:**

        br_template_list br_enroll_template(br_template tmpl)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    tmpl | br_template | Pointer to a [Template](../cpp_api/template/template.md).

* **output:** ([br_template_list](typedefs.md#br_template_list)) Returns a pointer to a [TemplateList](../cpp_api/templatelist/templatelist.md)

---

## br_enroll_template_list

Enroll a [TemplateList](../cpp_api/templatelist/templatelist.md) from the C API!

* **function definition:**

        void br_enroll_template_list(br_template_list tl)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    tl | [br_template_list](typedefs.md#br_template_list) | Pointer to a [TemplateList](../cpp_api/templatelist/templatelist.md)

* **output:** (void)

---

## br_compare_template_lists

Compare [TemplateLists](../cpp_api/templatelist/templatelist.md) from the C API!

* **function definition:**

        br_matrix_output br_compare_template_lists(br_template_list target, br_template_list query)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    target | br_template_list | Pointer to a [TemplateList](../cpp_api/templatelist/templatelist.md)
    query | br_template_list | Pointer to a [TemplateList](../cpp_api/templatelist/templatelist.md)


* **output:** ([br_matrix_output](typedefs.md#br_matrix_output)) Returns a pointer to a [MatrixOutput](../cpp_api/matrixoutput/matrixoutput.md)

---

## br_get_matrix_output_at

Get a value in a provided [MatrixOutput](../cpp_api/matrixoutput/matrixoutput.md).

* **function definition:**

        float br_get_matrix_output_at(br_matrix_output output, int row, int col)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    output | br_matrix_output | Pointer to [MatrixOutput](../cpp_api/matrixoutput/matrixoutput.md)
    row | int | Row index for lookup
    col | int | Column index for lookup

* **output:** (float) Returns the value of the [MatrixOutput](../cpp_api/matrixoutput/matrixoutput.md) at the provided indexes.

---

## br_get_template

Get a [Template](../cpp_api/template/template.md) from a [TemplateList](../cpp_api/templatelist/templatelist.md) at a specified index.

* **function definition:**

        br_template br_get_template(br_template_list tl, int index)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    tl | [br_template_list](typedefs.md#br_template_list) | Pointer to a [TemplateList](../cpp_api/templatelist/templatelist.md)
    index | int | Index into the template list. Should be in the range [0,len(tl) - 1].

* **output:** ([br_template](typedefs.md#br_template)) Returns a pointer to a [Template](../cpp_api/template/template.md)

---

## br_num_templates

Get the number of [Templates](../cpp_api/template/template.md) in a [TemplateList](../cpp_api/templatelist/templatelist.md).

* **function definition:**

        int br_num_templates(br_template_list tl)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    tl | [br_template_list](typedefs.md#br_template_list) | Pointer to a [TemplateList](../cpp_api/templatelist/templatelist.md)

* **output:** (int) Returns the size of the provided [TemplateList](../cpp_api/templatelist/templatelist.md)

---

## br_make_gallery

Initialize a [Gallery](../cpp_api/gallery/gallery.md) from a file.

* **function definition:**

        br_gallery br_make_gallery(const char *gallery)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    gallery | const char * | String location of gallery on disk.

* **output:** ([br_gallery](typedefs.md#br_gallery)) Returns a pointer to a [Gallery](../cpp_api/gallery/gallery.md) that has been created from the provided file

---

## br_load_from_gallery

Read a [TemplateList](../cpp_api/templatelist/templatelist.md) from a [Gallery](../cpp_api/gallery/gallery.md).

* **function definition:**

        br_template_list br_load_from_gallery(br_gallery gallery)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    gallery | [br_gallery](typedefs.md#br_gallery) | Pointer to a [Gallery](../cpp_api/gallery/gallery.md)

* **output:** ([br_template_list](typedefs.md#br_template_list)) Returns a pointer to a [TemplateList](../cpp_api/templatelist/templatelist.md) containing the data from the provided [Gallery](../cpp_api/gallery/gallery.md)

---

## br_add_template_to_gallery

Write a [Template](../cpp_api/template/template.md) to a [Gallery](../cpp_api/gallery/gallery.md)

* **function definition:**

        void br_add_template_to_gallery(br_gallery gallery, br_template tmpl)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    gallery | [br_gallery](typedefs.md#br_gallery) | Pointer to a [Gallery](../cpp_api/gallery/gallery.md)
    tmpl | [br_template](typedefs.md#br_template) | Pointer to a [Template](../cpp_api/template/template.md)

* **output:** (void)

---

## br_add_template_list_to_gallery

Write a [TemplateList](../cpp_api/templatelist/templatelist.md) to the [Gallery](../cpp_api/gallery/gallery.md) on disk.

* **function definition:**

        void br_add_template_list_to_gallery(br_gallery gallery, br_template_list tl)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    gallery | [br_gallery](typedefs.md#br_gallery) | Pointer to a [Gallery](../cpp_api/gallery/gallery.md)
    tl | [br_template_list](typedefs.md#br_template_list) | Pointer to a [TemplateList](../cpp_api/templatelist/templatelist.md)

* **output:** (void)

---

## br_close_gallery

Close a provided [Gallery](../cpp_api/gallery/gallery.md).

* **function definition:**

        void br_close_gallery(br_gallery gallery)

* **parameters:**

    Parameter | Type | Description
    --- | --- | ---
    gallery | [br_gallery](typedefs.md#br_gallery) | Pointer to a [Gallery](../cpp_api/gallery/gallery.md)

* **output:** (void)

[^1]: *Zhu et al.*
    **A Rank-Order Distance based Clustering Algorithm for Face Tagging**,
    CVPR 2011

<!-- Links -->
[R]: http://www.r-project.org/ "R"
[QRegExp]: http://doc.qt.io/qt-5/QRegExp.html "QRegExp"
[QString]: http://doc.qt.io/qt-5/QString.html "QString"
[QStringList]: http://doc.qt.io/qt-5/QStringList.html "QStringList"
[QPointF]: http://doc.qt.io/qt-5/QPointF.html "QPointF"
