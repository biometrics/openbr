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
#include "core/plot.h"
#include "core/qtutils.h"
#include "plugins/openbr_internal.h"
#include <opencv2/highgui/highgui.hpp>

using namespace br;

const char *br_about()
{
    static QByteArray about = Context::about().toLocal8Bit();
    return about.data();
}

void br_cat(int num_input_galleries, const char *input_galleries[], const char *output_gallery)
{
    Cat(QtUtils::toStringList(num_input_galleries, input_galleries), output_gallery);
}

void br_cluster(int num_simmats, const char *simmats[], float aggressiveness, const char *csv)
{
    ClusterGallery(QtUtils::toStringList(num_simmats, simmats), aggressiveness, csv);
}

void br_combine_masks(int num_input_masks, const char *input_masks[], const char *output_mask, const char *method)
{
    BEE::combineMasks(QtUtils::toStringList(num_input_masks, input_masks), output_mask, method);
}

void br_compare(const char *target_gallery, const char *query_gallery, const char *output)
{
    Compare(File(target_gallery), File(query_gallery), File(output));
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

float br_eval(const char *simmat, const char *mask, const char *csv)
{
    return Evaluate(simmat, mask, csv);
}

void br_eval_classification(const char *predicted_gallery, const char *truth_gallery, const char *predicted_property, const char *truth_property)
{
    EvalClassification(predicted_gallery, truth_gallery, predicted_property, truth_property);
}

void br_eval_clustering(const char *csv, const char *gallery)
{
    EvalClustering(csv, gallery);
}

float br_eval_detection(const char *predicted_gallery, const char *truth_gallery, const char *csv)
{
    return EvalDetection(predicted_gallery, truth_gallery, csv);
}

float br_eval_landmarking(const char *predicted_gallery, const char *truth_gallery, const char *csv, int normalization_index_a, int normalization_index_b)
{
    return EvalLandmarking(predicted_gallery, truth_gallery, csv, normalization_index_a, normalization_index_b);
}

void br_eval_regression(const char *predicted_gallery, const char *truth_gallery, const char *predicted_property, const char *truth_property)
{
    EvalRegression(predicted_gallery, truth_gallery, predicted_property, truth_property);
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

void br_initialize(int &argc, char *argv[], const char *sdk_path)
{
    Context::initialize(argc, argv, sdk_path);
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

const char *br_most_recent_message()
{
    static QByteArray byteArray;
    byteArray = Globals->mostRecentMessage.toLocal8Bit();
    return byteArray.data();
}

const char *br_objects(const char *abstractions, const char *implementations, bool parameters)
{
    static QByteArray objects;

    QStringList objectList;
    QRegExp abstractionsRegExp(abstractions);
    QRegExp implementationsRegExp(implementations);

    if (abstractionsRegExp.exactMatch("Abbreviation"))
        foreach (const QString &name, Globals->abbreviations.keys())
            if (implementationsRegExp.exactMatch(name))
                objectList.append(name + (parameters ? "\t" + Globals->abbreviations[name] : ""));

    if (abstractionsRegExp.exactMatch("Distance"))
        foreach (const QString &name, Factory<Distance>::names())
            if (implementationsRegExp.exactMatch(name))
                objectList.append(name + (parameters ? "\t" + Factory<Distance>::parameters(name) : ""));

    if (abstractionsRegExp.exactMatch("Format"))
        foreach (const QString &name, Factory<Format>::names())
            if (implementationsRegExp.exactMatch(name))
                objectList.append(name + (parameters ? "\t" + Factory<Format>::parameters(name) : ""));

    if (abstractionsRegExp.exactMatch("Initializer"))
        foreach (const QString &name, Factory<Initializer>::names())
            if (implementationsRegExp.exactMatch(name))
                objectList.append(name + (parameters ? "\t" + Factory<Initializer>::parameters(name) : ""));

    if (abstractionsRegExp.exactMatch("Output"))
        foreach (const QString &name, Factory<Output>::names())
            if (implementationsRegExp.exactMatch(name))
                objectList.append(name + (parameters ? "\t" + Factory<Output>::parameters(name) : ""));

    if (abstractionsRegExp.exactMatch("Transform"))
        foreach (const QString &name, Factory<Transform>::names())
            if (implementationsRegExp.exactMatch(name))
                objectList.append(name + (parameters ? "\t" + Factory<Transform>::parameters(name) : ""));

    objects = objectList.join("\n").toLocal8Bit();
    return objects.data();
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

const char *br_scratch_path()
{
    static QByteArray byteArray;
    byteArray = Context::scratchPath().toLocal8Bit();
    return byteArray.data();
}

const char *br_sdk_path()
{
    static QByteArray sdkPath = QDir(Globals->sdkPath).absolutePath().toLocal8Bit();
    return sdkPath.data();
}

void br_get_header(const char *matrix, const char **target_gallery, const char **query_gallery)
{
    static QByteArray targetGalleryData, queryGalleryData;
    QString targetGalleryString, queryGalleryString;
    BEE::readMatrixHeader(matrix, &targetGalleryString, &queryGalleryString);
    targetGalleryData = targetGalleryString.toLatin1();
    queryGalleryData = queryGalleryString.toLatin1();
    *target_gallery = targetGalleryData.data();
    *query_gallery = queryGalleryData.data();
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
    static QByteArray version = Context::version().toLocal8Bit();
    return version.data();
}

void br_slave_process(const char * baseName)
{
    WorkerProcess * worker = new WorkerProcess;
    worker->transform = Globals->algorithm;
    worker->baseName = baseName;
    worker->mainLoop();
    delete worker;
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

br_template_list br_enroll_template(br_template tmpl)
{
    Template *t = reinterpret_cast<Template*>(tmpl);
    TemplateList *tl = new TemplateList();
    tl->append(*t);
    Enroll(*tl);
    return (br_template_list)tl;
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
