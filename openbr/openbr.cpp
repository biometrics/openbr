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

#include <openbr/openbr_plugin.h>

#include "core/bee.h"
#include "core/cluster.h"
#include "core/eval.h"
#include "core/fuse.h"
#include "core/likely.h"
#include "core/plot.h"
#include "core/qtutils.h"
#include "plugins/openbr_internal.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>

using namespace br;

static int partialCopy(const QString &string, char *buffer, int buffer_length)
{
    const QByteArray byteArray = string.toLocal8Bit();

    int copyLength = std::min(buffer_length-1, byteArray.size());
    if (copyLength < 0)
        return byteArray.size() + 1;

    memcpy(buffer, byteArray.constData(), copyLength);
    buffer[copyLength] = '\0';

    return byteArray.size() + 1;
}

const char *br_about()
{
    static const QByteArray about = Context::about().toLocal8Bit();
    return about.constData();
}

void br_cat(int num_input_galleries, const char *input_galleries[], const char *output_gallery)
{
    Cat(QtUtils::toStringList(num_input_galleries, input_galleries), output_gallery);
}

void br_cluster(int num_simmats, const char *simmats[], float aggressiveness, const char *csv)
{
    ClusterSimmat(QtUtils::toStringList(num_simmats, simmats), aggressiveness, csv);
}

void br_combine_masks(int num_input_masks, const char *input_masks[], const char *output_mask, const char *method)
{
    BEE::combineMasks(QtUtils::toStringList(num_input_masks, input_masks), output_mask, method);
}

void br_compare(const char *target_gallery, const char *query_gallery, const char *output)
{
    Compare(File(target_gallery), File(query_gallery), File(output));
}

void br_compare_n(int num_targets, const char *target_galleries[], const char *query_gallery, const char *output)
{
    if (num_targets > 1) Compare(QtUtils::toStringList(num_targets, target_galleries).join(";")+"(separator=;)", File(query_gallery), File(output));
    else                 Compare(File(target_galleries[0]), File(query_gallery), File(output));
}

void br_pairwise_compare(const char *target_gallery, const char *query_gallery, const char *output)
{
    PairwiseCompare(File(target_gallery), File(query_gallery), File(output));
}

void br_convert(const char *file_type, const char *input_file, const char *output_file)
{
    Convert(File(file_type), File(input_file), File(output_file));
}

void br_enroll(const char *input, const char *gallery)
{
    Enroll(File(input), File(gallery));
}

void br_enroll_n(int num_inputs, const char *inputs[], const char *gallery)
{
    if (num_inputs > 1) Enroll(QtUtils::toStringList(num_inputs, inputs).join(";")+"(separator=;)", File(gallery));
    else                Enroll(File(inputs[0]), gallery);
}

void br_project(const char *input, const char *gallery)
{
    Project(File(input), File(gallery));
}

float br_eval(const char *simmat, const char *mask, const char *csv, int matches)
{
    return Evaluate(simmat, mask, csv, matches);
}

void br_assert_eval(const char *simmat, const char *mask, const float accuracy)
{
    assertEval(simmat, mask, accuracy);
}

float br_inplace_eval(const char *simmat, const char *target, const char *query, const char *csv)
{
    return InplaceEval(simmat, target, query, csv);
}

void br_eval_classification(const char *predicted_gallery, const char *truth_gallery, const char *predicted_property, const char *truth_property)
{
    EvalClassification(predicted_gallery, truth_gallery, predicted_property, truth_property);
}

void br_eval_clustering(const char *clusters, const char *truth_gallery, const char *truth_property, bool cluster_csv, const char *cluster_property)
{
    EvalClustering(clusters, truth_gallery, truth_property, cluster_csv, cluster_property);
}

float br_eval_detection(const char *predicted_gallery, const char *truth_gallery, const char *csv, bool normalize, int minSize, int maxSize, float relativeMinSize)
{
    return EvalDetection(predicted_gallery, truth_gallery, csv, normalize, minSize, maxSize, relativeMinSize);
}

float br_eval_landmarking(const char *predicted_gallery, const char *truth_gallery, const char *csv, int normalization_index_a, int normalization_index_b, int sample_index, int total_examples)
{
    return EvalLandmarking(predicted_gallery, truth_gallery, csv, normalization_index_a, normalization_index_b, sample_index, total_examples);
}

void br_eval_regression(const char *predicted_gallery, const char *truth_gallery, const char *predicted_property, const char *truth_property)
{
    EvalRegression(predicted_gallery, truth_gallery, predicted_property, truth_property);
}

void br_eval_knn(const char *knnGraph, const char *knnTruth, const char *csv)
{
    EvalKNN(knnGraph, knnTruth, csv);
}

void br_eval_eer(const char *predicted_xml, const char *gt_property, const char *distribution_property, const char *pdf)
{
    EvalEER(predicted_xml, gt_property, distribution_property, pdf);
}

void br_finalize()
{
    Context::finalize();
}

void br_fuse(int num_input_simmats, const char *input_simmats[],
             const char *normalization, const char *fusion, const char *output_simmat)
{
    Fuse(QtUtils::toStringList(num_input_simmats, input_simmats), normalization, fusion, output_simmat);
}

void br_initialize(int &argc, char *argv[], const char *sdk_path, bool use_gui)
{
    Context::initialize(argc, argv, sdk_path, use_gui);
    if (!Globals)
        abort();
}

void br_initialize_default()
{
    int argc = 1;
    char app[] = "br";
    char *argv[1] = {app};
    Context::initialize(argc, argv, "");
}

bool br_is_classifier(const char *algorithm)
{
    return IsClassifier(algorithm);
}

void br_make_mask(const char *target_input, const char *query_input, const char *mask)
{
    BEE::makeMask(target_input, query_input, mask);
}

void br_make_pairwise_mask(const char *target_input, const char *query_input, const char *mask)
{
    BEE::makePairwiseMask(target_input, query_input, mask);
}

int br_most_recent_message(char *buffer, int buffer_length)
{
    return partialCopy(Globals->mostRecentMessage, buffer, buffer_length);
}

int br_objects(char *buffer, int buffer_length, const char *abstractions, const char *implementations, bool parameters)
{
    return partialCopy(br::Context::objects(abstractions, implementations, parameters).join('\n'), buffer, buffer_length);
}

bool br_plot(int num_files, const char *files[], const char *destination, bool show)
{
    return Plot(QtUtils::toStringList(num_files, files), destination, show);
}

bool br_plot_detection(int num_files, const char *files[], const char *destination, bool show)
{
    return PlotDetection(QtUtils::toStringList(num_files, files), destination, show);
}

bool br_plot_landmarking(int num_files, const char *files[], const char *destination, bool show)
{
    return PlotLandmarking(QtUtils::toStringList(num_files, files), destination, show);
}

bool br_plot_metadata(int num_files, const char *files[], const char *columns, bool show)
{
    return PlotMetadata(QtUtils::toStringList(num_files, files), columns, show);
}

bool br_plot_knn(int num_files, const char *files[], const char *destination, bool show)
{
    return PlotKNN(QtUtils::toStringList(num_files, files), destination, show);
}

bool br_plot_eer(int num_files, const char *files[], const char *destination, bool show)
{
    return PlotEER(QtUtils::toStringList(num_files, files), destination, show);
}

float br_progress()
{
    return Globals->progress();
}

void br_read_pipe(const char *pipe, int *argc, char ***argv)
{
    static QList<QByteArray> byteArrayList;
    static QVector<char*> rawCharArrayList;

    QFile file(pipe);
    file.open(QFile::ReadOnly);
    QTextStream stream(&file);

    QStringList args;
    while (args.isEmpty()) {
        args = QtUtils::parse(stream.readAll(), ' ');
        if (args.isEmpty()) QThread::sleep(100);
    }

    file.close();

    byteArrayList.clear(); rawCharArrayList.clear();
    foreach (const QString &string, args) {
        byteArrayList.append(string.toLocal8Bit());
        rawCharArrayList.append(byteArrayList.last().data());
    }

    *argc = byteArrayList.size();
    *argv = rawCharArrayList.data();
}

int br_scratch_path(char *buffer, int buffer_length)
{
    return partialCopy(Context::scratchPath(), buffer, buffer_length);
}

const char *br_sdk_path()
{
    static const QByteArray sdkPath = QDir(Globals->sdkPath).absolutePath().toLocal8Bit();
    return sdkPath.constData();
}

void br_get_header(const char *matrix, const char **target_gallery, const char **query_gallery)
{
    static QByteArray targetGalleryData, queryGalleryData;
    QString targetGalleryString, queryGalleryString;
    BEE::readMatrixHeader(matrix, &targetGalleryString, &queryGalleryString);
    targetGalleryData = targetGalleryString.toLatin1();
    queryGalleryData = queryGalleryString.toLatin1();
    *target_gallery = targetGalleryData.constData();
    *query_gallery = queryGalleryData.constData();
}

void br_set_header(const char *matrix, const char *target_gallery, const char *query_gallery)
{
    BEE::writeMatrixHeader(matrix, target_gallery, query_gallery);
}

void br_set_property(const char *key, const char *value)
{
    Globals->setProperty(key, value);
}

int br_time_remaining()
{
    return Globals->timeRemaining();
}

void br_train(const char *input, const char *model)
{
    Train(input, model);
}

void br_train_n(int num_inputs, const char *inputs[], const char *model)
{
    if (num_inputs > 1) Train(QtUtils::toStringList(num_inputs, inputs).join(";")+"(separator=;)", File(model));
    else                Train(File(inputs[0]), model);
}

const char *br_version()
{
    static const QByteArray version = Context::version().toLocal8Bit();
    return version.data();
}

void br_slave_process(const char *baseName)
{
#ifdef BR_WITH_QTNETWORK
    WorkerProcess *worker = new WorkerProcess;
    worker->transform = Globals->algorithm;
    worker->baseName = baseName;
    worker->mainLoop();
    delete worker;
#else
    (void) baseName;
    qFatal("multiprocess support requires building with QtNetwork enabled (set BR_WITH_QTNETWORK in cmake).");
#endif
}

br_template br_load_img(const char *data, int len)
{
    std::vector<char> buf(data, data+len);
    cv::Mat img = cv::imdecode(cv::Mat(buf), CV_LOAD_IMAGE_COLOR);
    Template *tmpl = new Template(img);
    return (br_template)tmpl;
}

unsigned char *br_unload_img(br_template tmpl)
{
    Template *t = reinterpret_cast<Template*>(tmpl);
    return t->m().data;
}

br_template_list br_template_list_from_buffer(const char *buf, int len)
{
    QByteArray arr(buf, len);
    TemplateList *tl = new TemplateList();
    *tl = TemplateList::fromBuffer(arr);
    return (br_template_list)tl;
}

void br_free_template(br_template tmpl)
{
    Template *t = reinterpret_cast<Template*>(tmpl);
    delete t;
}

void br_free_template_list(br_template_list tl)
{
    TemplateList *realTL = reinterpret_cast<TemplateList*>(tl);
    delete realTL;
}

void br_free_output(br_matrix_output output)
{
    MatrixOutput *matOut = reinterpret_cast<MatrixOutput*>(output);
    delete matOut;
}

int br_img_rows(br_template tmpl)
{
    Template *t = reinterpret_cast<Template*>(tmpl);
    return t->m().rows;
}

int br_img_cols(br_template tmpl)
{
    Template *t = reinterpret_cast<Template*>(tmpl);
    return t->m().cols;
}

int br_img_channels(br_template tmpl)
{
    Template *t = reinterpret_cast<Template*>(tmpl);
    return t->m().channels();
}

bool br_img_is_empty(br_template tmpl)
{
    Template *t = reinterpret_cast<Template*>(tmpl);
    return t->m().empty();
}

int br_get_filename(char *buffer, int buffer_length, br_template tmpl)
{
    return partialCopy(reinterpret_cast<Template*>(tmpl)->file.name, buffer, buffer_length);
}

void br_set_filename(br_template tmpl, const char *filename)
{
    Template *t = reinterpret_cast<Template*>(tmpl);
    t->file.name = filename;
}

int br_get_metadata_string(char *buffer, int buffer_length, br_template tmpl, const char *key)
{
    Template *t = reinterpret_cast<Template*>(tmpl);
    QVariant qvar = t->file.value(key);
    return partialCopy(QtUtils::toString(qvar), buffer, buffer_length);
}

br_template_list br_enroll_template(br_template tmpl)
{
    Template *t = reinterpret_cast<Template*>(tmpl);
    TemplateList *tl = new TemplateList();
    tl->append(*t);
    Enroll(*tl);
    return (br_template_list)tl;
}

void br_enroll_template_list(br_template_list tl)
{
    TemplateList *realTL = reinterpret_cast<TemplateList*>(tl);
    Enroll(*realTL);
}

br_matrix_output br_compare_template_lists(br_template_list target, br_template_list query)
{
    TemplateList *targetTL = reinterpret_cast<TemplateList*>(target);
    TemplateList *queryTL = reinterpret_cast<TemplateList*>(query);
    MatrixOutput *output = MatrixOutput::make(targetTL->files(), queryTL->files());
    CompareTemplateLists(*targetTL, *queryTL, output);
    return (br_matrix_output)output;
}

float br_get_matrix_output_at(br_matrix_output output, int row, int col)
{
    MatrixOutput *matOut = reinterpret_cast<MatrixOutput*>(output);
    return matOut->data.at<float>(row, col);
}

br_template br_get_template(br_template_list tl, int index)
{
    TemplateList *realTL = reinterpret_cast<TemplateList*>(tl);
    return (br_template)&realTL->at(index);
}

int br_num_templates(br_template_list tl)
{
    TemplateList *realTL = reinterpret_cast<TemplateList*>(tl);
    return realTL->size();
}

br_gallery br_make_gallery(const char *gallery)
{
    Gallery *gal = Gallery::make(File(gallery));
    return (br_gallery)gal;
}

br_template_list br_load_from_gallery(br_gallery gallery)
{
    Gallery *gal = reinterpret_cast<Gallery*>(gallery);
    TemplateList *tl = new TemplateList();
    *tl = gal->read();
    return (br_template_list)tl;
}

void br_add_template_to_gallery(br_gallery gallery, br_template tmpl)
{
    Gallery *gal = reinterpret_cast<Gallery*>(gallery);
    Template *t = reinterpret_cast<Template*>(tmpl);
    gal->write(*t);
}

void br_add_template_list_to_gallery(br_gallery gallery, br_template_list tl)
{
    Gallery *gal = reinterpret_cast<Gallery*>(gallery);
    TemplateList *realTL = reinterpret_cast<TemplateList*>(tl);
    gal->writeBlock(*realTL);
}

void br_close_gallery(br_gallery gallery)
{
    Gallery *gal = reinterpret_cast<Gallery*>(gallery);
    delete gal;
}

void br_deduplicate(const char *input_gallery, const char *output_gallery, const char *threshold)
{
    br::Deduplicate(input_gallery, output_gallery, threshold);
}

void br_likely(const char *input_type, const char *output_type, const char *output_source_file)
{
    br::Likely(input_type, output_type, output_source_file);
}
