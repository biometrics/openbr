/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2012 The MITRE Corporation                                      *
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License");           *
 * you may not use this file except in compliance with the License.          *
 * You may obtain a copy of the License at                                   *
 *                                                                           *
 *     http://www.apache.org/licenses/LICENSE-2.0                            *
 *                                                                           *
 * Unless required by applicable law or agreed to in writing, software       *
 * distributed under the License is distributed on an "AS IS" BASIS,         *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 * See the License for the specific language governing permissions and       *
 * limitations under the License.                                            *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef OPENBR_H
#define OPENBR_H

#include <openbr/openbr_export.h>

#ifdef __cplusplus
extern "C" {
#endif


BR_EXPORT const char *br_about();

BR_EXPORT void br_cat(int num_input_galleries, const char *input_galleries[], const char *output_gallery);

BR_EXPORT void br_deduplicate(const char *input_gallery, const char *output_gallery, const char *threshold);

BR_EXPORT void br_cluster(int num_simmats, const char *simmats[], float aggressiveness, const char *csv);

BR_EXPORT void br_combine_masks(int num_input_masks, const char *input_masks[], const char *output_mask, const char *method);

BR_EXPORT void br_compare(const char *target_gallery, const char *query_gallery, const char *output = "");

BR_EXPORT void br_compare_n(int num_targets, const char *target_galleries[], const char *query_gallery, const char *output);

BR_EXPORT void br_pairwise_compare(const char *target_gallery, const char *query_gallery, const char *output = "");

BR_EXPORT void br_convert(const char *file_type, const char *input_file, const char *output_file);

BR_EXPORT void br_enroll(const char *input, const char *gallery = "");

BR_EXPORT void br_enroll_n(int num_inputs, const char *inputs[], const char *gallery = "");

BR_EXPORT void br_project(const char *input, const char *output);

BR_EXPORT float br_eval(const char *simmat, const char *mask, const char *csv = "", int matches = 0);

BR_EXPORT void br_assert_eval(const char *simmat, const char *mask, const float accuracy);

BR_EXPORT float br_inplace_eval(const char * simmat, const char *target, const char *query, const char *csv = "");

BR_EXPORT void br_eval_classification(const char *predicted_gallery, const char *truth_gallery, const char *predicted_property = "", const char *truth_property = "");

BR_EXPORT void br_eval_clustering(const char *clusters, const char *truth_gallery, const char *truth_property = "", bool cluster_csv = true, const char *cluster_property = "");

BR_EXPORT float br_eval_detection(const char *predicted_gallery, const char *truth_gallery, const char *csv = "", bool normalize = false, int minSize = 0, int maxSize = 0, float relativeMinSize = 0);

BR_EXPORT float br_eval_landmarking(const char *predicted_gallery, const char *truth_gallery, const char *csv = "", int normalization_index_a = 0, int normalization_index_b = 1, int sample_index = 0, int total_examples = 5);

BR_EXPORT void br_eval_regression(const char *predicted_gallery, const char *truth_gallery, const char *predicted_property = "", const char *truth_property = "");

BR_EXPORT void br_eval_knn(const char *knnGraph, const char *knnTruth, const char *csv = "");

BR_EXPORT void br_eval_eer(const char *predicted_xml, const char *gt_property = "", const char *distribution_property = "", const char *pdf = "");

BR_EXPORT void br_finalize();

BR_EXPORT void br_fuse(int num_input_simmats, const char *input_simmats[],
                       const char *normalization, const char *fusion, const char *output_simmat);

BR_EXPORT void br_initialize(int &argc, char *argv[], const char *sdk_path = "", bool use_gui = false);

BR_EXPORT void br_initialize_default();

BR_EXPORT bool br_is_classifier(const char *algorithm);

BR_EXPORT void br_make_mask(const char *target_input, const char *query_input, const char *mask);

BR_EXPORT void br_make_pairwise_mask(const char *target_input, const char *query_input, const char *mask);

BR_EXPORT int br_most_recent_message(char * buffer, int buffer_length);

BR_EXPORT int br_objects(char * buffer, int buffer_length, const char *abstractions = ".*", const char *implementations = ".*", bool parameters = true);

BR_EXPORT bool br_plot(int num_files, const char *files[], const char *destination, bool show = false);

BR_EXPORT bool br_plot_detection(int num_files, const char *files[], const char *destination, bool show = false);

BR_EXPORT bool br_plot_landmarking(int num_files, const char *files[], const char *destination, bool show = false);

BR_EXPORT bool br_plot_metadata(int num_files, const char *files[], const char *columns, bool show = false);

BR_EXPORT bool br_plot_knn(int num_files, const char *files[], const char *destination, bool show = false);

BR_EXPORT bool br_plot_eer(int num_files, const char *files[], const char *destination, bool show = false);

BR_EXPORT float br_progress();

BR_EXPORT void br_read_pipe(const char *pipe, int *argc, char ***argv);

BR_EXPORT int br_scratch_path(char * buffer, int buffer_length);

BR_EXPORT const char *br_sdk_path();

BR_EXPORT void br_get_header(const char *matrix, const char **target_gallery, const char **query_gallery);

BR_EXPORT void br_set_header(const char *matrix, const char *target_gallery, const char *query_gallery);

BR_EXPORT void br_set_property(const char *key, const char *value);

BR_EXPORT int br_time_remaining();

BR_EXPORT void br_train(const char *input, const char *model = "");

BR_EXPORT void br_train_n(int num_inputs, const char *inputs[], const char *model = "");

BR_EXPORT const char *br_version();

BR_EXPORT void br_slave_process(const char * baseKey);

BR_EXPORT void br_likely(const char *input_type, const char *output_type, const char *output_source_file);

// to avoid having to include unwanted headers
// this will be this header's conception of a Template
// any functions that need a Template pointer
// will take this typedef and cast it
typedef void* br_template;
typedef void* br_template_list;
typedef void* br_gallery;
typedef void* br_matrix_output;

BR_EXPORT br_template br_load_img(const char *data, int len);

BR_EXPORT unsigned char* br_unload_img(br_template tmpl);

BR_EXPORT br_template_list br_template_list_from_buffer(const char *buf, int len);

BR_EXPORT void br_free_template(br_template tmpl);

BR_EXPORT void br_free_template_list(br_template_list tl);

BR_EXPORT void br_free_output(br_matrix_output output);

BR_EXPORT int br_img_rows(br_template tmpl);

BR_EXPORT int br_img_cols(br_template tmpl);

BR_EXPORT int br_img_channels(br_template tmpl);

BR_EXPORT bool br_img_is_empty(br_template tmpl);

BR_EXPORT int br_get_filename(char * buffer, int buffer_length, br_template tmpl);

BR_EXPORT void br_set_filename(br_template tmpl, const char *filename);

BR_EXPORT int br_get_metadata_string(char * buffer, int buffer_length, br_template tmpl, const char *key);

BR_EXPORT br_template_list br_enroll_template(br_template tmpl);

BR_EXPORT void br_enroll_template_list(br_template_list tl);

BR_EXPORT br_matrix_output br_compare_template_lists(br_template_list target, br_template_list query);

BR_EXPORT float br_get_matrix_output_at(br_matrix_output output, int row, int col);

BR_EXPORT br_template br_get_template(br_template_list tl, int index);

BR_EXPORT int br_num_templates(br_template_list tl);

BR_EXPORT br_gallery br_make_gallery(const char *gallery);

BR_EXPORT br_template_list br_load_from_gallery(br_gallery gallery);

BR_EXPORT void br_add_template_to_gallery(br_gallery gallery, br_template tmpl);

BR_EXPORT void br_add_template_list_to_gallery(br_gallery gallery, br_template_list tl);

BR_EXPORT void br_close_gallery(br_gallery gallery);



#ifdef __cplusplus
}
#endif

#endif // OPENBR_H
