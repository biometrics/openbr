The command line API is a tool to run OpenBR from the command line. The command line is the easiest and fastest way to run OpenBR!

The following is a detailed description of the command line API. The command line API is really just a set of wrappers to call the [C API](c_api.md). All of the flags in this API have a corresponding C API call. To help display the examples the following shorthand definitions will be used:

Shortcut | Definition
--- | ---
&lt;arg&gt; | &lt;&gt; Represent an input argument
\{arg\} | \{\} Represent an output argument
[arg] | [] Represent an optional argument
(arg0 &#124; ... &#124; argN) | (... &#124; ...) Represent a choice.

## Algorithms

Almost every command line process needs to specify an algorithm to work properly. Algorithms in OpenBR are described in detail [here](../tutorials.md#algorithms-in-openbr). To specify algorithms to the command line use the <tt>-algorithm</tt> flag like so:

        -algorithm "AlgorithmString"

Make sure you use the quotes if your algorithm is longer than one plugin because special characters in OpenBR are also special characters (with very different meanings!) in Bash.

## Core Commands

### -train {: #train }

Train a model

* **arguments:**

        -train <gallery> ... <gallery> [{model}]

* **wraps:** [br_train_n](c_api/functions.md#br_train_n)

### -enroll {: #enroll }

Enroll a [Gallery](cpp_api/gallery/gallery.md) through an algorithm

* **arguments:**

        -enroll <input_gallery> ... <input_gallery> {output_gallery}

* **wraps:** [br_enroll](c_api/functions.md#br_enroll) or [br_enroll_n](c_api/functions.md#br_enroll_n) depending on the input size

### -compare {: #compare }

Compare query [Templates](cpp_api/template/template.md) against a target [Gallery](cpp_api/gallery/gallery.md)

* **arguments:**

        -compare <target_gallery> <query_gallery> [{output}]

* **wraps:** [br_compare](c_api/functions.md#br_compare)

### -pairwiseCompare {: #pairwisecompare }

DOCUMENT ME

* **arguments:**

        -pairwiseCompare <target_gallery> <query_gallery> [{output}]

* **wraps:** [br_pairwise_compare](c_api/functions.md#br_pairwise_compare)

### -eval {: #eval }

Evaluate a similarity matrix

* **arguments:**

        -eval <simmat> [<mask>] [{csv}] [{matches}]

* **wraps:** [br_eval](c_api/functions.md#br_eval)

### -inplaceEval {: #inplaceeval }

DOCUMENT ME

* **arguments:**

        -inplaceEval <simmat> <target> <query> [{output}]

* **wraps:** [br_inplace_eval](c_api/functions.md#br_inplace_eval)

### -plot {: #inplaceeval}

Plot the results of an evaluation

* **arguments:**

               -plot <file> ... <file> {destination}

* **wraps:** [br_plot](c_api/functions.md#br_plot)


## Other Commands

### -fuse {: #fuse }

Perform score level fusion on similarity matrices.

* **arguments:**

        -fuse <simmat> ... <simmat> (None|MinMax|ZScore|WScore) (Min|Max|Sum[W1:W2:...:Wn]|Replace|Difference|None) {simmat}

* **wraps:** [br_fuse](c_api/functions.md#br_fuse)

### -cluster {: #cluster }

Clusters one or more similarity matrices into a list of subjects

* **arguments:**

        -cluster <simmat> ... <simmat> <aggressiveness> {csv}

* **wraps:** [br_cluster](c_api/functions.md#br_cluster)

### -makeMask {: #makemask }

Constructs a mask from target and query inputs

* **arguments:**

        -makeMask <target_gallery> <query_gallery> {mask}

* **wraps:** [br_make_mask](c_api/functions.md#br_make_mask)

### -makePairwiseMask {: #makepairwisemask }

Constructs a mask from target and query inputs considering the target and input sets to be definite pairwise comparisons.

* **arguments:**

        -makePairwiseMask <target_gallery> <query_gallery> {mask}

* **wraps:** [br_make_pairwise_mask](c_api/functions.md#br_make_pairwise_mask)

### -combineMasks {: #combinemask }

Combines several equal-sized mask matrices. A comparison may not be simultaneously indentified as both a genuine and an imposter by different input masks.

* **arguments:**

        -combineMasks <mask> ... <mask> {mask} (And|Or)

* **wraps:** [br_combine_masks](c_api/functions.md#br_combine_masks)

### -cat {: #cat }

Concatenates a list of galleries into 1 gallery

* **arguments:**

        -cat <gallery> ... <gallery> {gallery}

* **wraps:** [br_cat](c_api/functions.md#br_cat)

### -convert {: #convert }

Convert a file to a different type. Files can only be converted to types within the same group. For example formats can only be converted to other formats.

* **arguments:**

        -convert (Format|Gallery|Output) <input_file> {output_file}

* **wraps:** [br_convert](c_api/functions.md#br_convert)

### -evalClassification {: #evalclassification }

Evaluates and prints classification accuracy to terminal

* **arguments:**

        -evalClassification <predicted_gallery> <truth_gallery> <predicted property name> <ground truth property name>

* **wraps:** [br_eval_classification](c_api/functions.md#br_eval_classification)

### -evalClustering {: #evalclustering }

Evaluates and prints clustering accuracy to the terminal

* **arguments:**

        -evalClustering <clusters> <gallery>

* **wraps:** [br_eval_clustering](c_api/functions.md#br_eval_clustering)

### -evalDetection {: #evaldetection }

Evaluates and prints detection accuracy to terminal

* **arguments:**

        -evalDetection <predicted_gallery> <truth_gallery> [{csv}] [{normalize}] [{minSize}] [{maxSize}]

* **wraps:** [br_eval_detection](c_api/functions.md#br_eval_detection)

### -evalLandmarking {: #evallandmarking }

Evaluates and prints landmarking accuracy to terminal

* **arguments:**

        -evalLandmarking <predicted_gallery> <truth_gallery> [{csv} [<normalization_index_a> <normalization_index_b>] [sample_index] [total_examples]]

* **wraps:** [br_eval_landmarking](c_api/functions.md#br_eval_landmarking)


### -evalRegression {: #evalregression }

Evaluates regression accuracy to disk

* **arguments:**

        -evalRegression <predicted_gallery> <truth_gallery> <predicted property name> <ground truth property name>

* **wraps:** [br_eval_regression](c_api/functions.md#br_eval_regression)

### -assertEval {: #asserteval }

Evaluates the similarity matrix using the mask matrix.  Function aborts if TAR @ FAR = 0.001 does not meet an expected performance value.

* **arguments:**

        -assertEval <simmat> <mask> <accuracy>

* **wraps:** [br_assert_eval](c_api/functions.md#br_assert_eval)

### -plotDetection {: #plotdetection }

Renders detection performance figures for a set of .csv files created by [-evalDetection](#evaldetection).

* **arguments:**

        -plotDetection <file> ... <file> {destination}

* **wraps:** [br_plot_detection](c_api/functions.md#br_plot_detection)

### -plotLandmarking {: #plotlandmarking }

Renders landmarking performance figures for a set of .csv files created by [-evalLandmarking](#evallandmarking)

* **arguments:**

        -plotLandmarking <file> ... <file> {destination}

* **wraps:** [br_plot_landmarking](c_api/functions.md#br_plot_landmarking)

### -plotMetadata {: #plotmetadata }

Renders metadata figures for a set of .csv files with specified columns

* **arguments:**

        -plotMetadata <file> ... <file> <columns>

* **wraps:** [br_plot_metadata](c_api/functions.md#br_plot_metadata)

### -project {: #project }

A naive alternative to [-enroll](#enroll)

* **arguments:**

        -project <input_gallery> {output_gallery}

* **wraps:** [br_project](c_api/functions.md#br_project)

### -getHeader {: #getheader }

Retrieve the target and query inputs in the [BEE matrix](../tutorials.md#the-evaluation-harness) header

* **arguments:**

        -getHeader <matrix>

* **wraps:** [br_get_header](c_api/functions.md#br_get_header)

### -setHeader {: #setheader }

Update the target and query inputs in the [BEE matrix](../tutorials.md#the-evaluation-harness) header

* **arguments:**

        -setHeader {<matrix>} <target_gallery> <query_gallery>

* **wraps:** [br_set_header](c_api/functions.md#br_set_header)

### -&lt;key&gt; &lt;value&gt; {: #setproperty }

Appends a provided value to the [global metadata](cpp_api/context/context.md) using a provided key

* **arguments:**

        -<key> <value>

* **wraps:** [br_set_property](c_api/functions.md#br_set_property)


## Miscellaneous

### -help {: #help }

Print command line API documentation to the terminal

* **arguments:**

        -help

* **wraps:** N/A

### -gui {: #gui }

If this flag is set OpenBR will enable GUI windows to be launched. It must be the first flag set.

* **arguments:**

        br -gui

* **wraps:** N/A

### -objects {: #objects }

Returns names and parameters for the requested objects. Each object is newline separated. Arguments are separated from the object name with a tab. This function uses [QRegExp][QRegExp] syntax

* **arguments:**

        -objects [abstraction [implementation]]

* **wraps:** [br_objects](c_api/functions.md#br_objects)

### -about {: #about }

Get a string with the name, version, and copyright of the project. This string is suitable for printing or terminal

* **arguments:**

        -about

* **wraps:** [br_about](c_api/functions.md#br_about)

### -version {: #version }

Get the current OpenBR version

* **arguments:**

        -version

* **wraps:** [br_version](c_api/functions.md#br_version)

### -slave {: #slave }

For internal use via [ProcessWrapperTransform](../plugin_docs/core.md#processwrappertransform)

* **arguments:**

        -slave <baseKey>

* **wraps:** [br_slave_process](c_api/functions.md#br_slave_process)

### -daemon {: #daemon }

DOCUMENT ME

* **arguments:**

        -daemon <daemon_pipe>

* **wraps:** N/A

### -exit

Exit the application

* **arguments:**

        -exit

* **wraps:** N/A

<!-- Link -->
[QRegExp]: http://doc.qt.io/qt-5/QRegExp.html "QRegExp"
