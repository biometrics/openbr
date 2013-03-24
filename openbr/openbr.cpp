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
#include "core/classify.h"
#include "core/cluster.h"
#include "core/fuse.h"
#include "core/plot.h"
#include "core/qtutils.h"

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

void br_confusion(const char *file, float score, int *true_positives, int *false_positives, int *true_negatives, int *false_negatives)
{
    return Confusion(file, score, *true_positives, *false_positives, *true_negatives, *false_negatives);
}

void br_convert(const char *input, const char *output)
{
    Convert(File(input), File(output));
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

void br_eval_classification(const char *predicted_input, const char *truth_input)
{
    EvalClassification(predicted_input, truth_input);
}

void br_eval_clustering(const char *csv, const char *input)
{
    EvalClustering(csv, input);
}

void br_eval_regression(const char *predicted_input, const char *truth_input)
{
    EvalRegression(predicted_input, truth_input);
}

void br_finalize()
{
    Context::finalize();
}

void br_fuse(int num_input_simmats, const char *input_simmats[], const char *mask,
             const char *normalization, const char *fusion, const char *output_simmat)
{
    Fuse(QtUtils::toStringList(num_input_simmats, input_simmats), mask, normalization, fusion, output_simmat);
}

void br_initialize(int &argc, char *argv[], const char *sdk_path, bool gui)
{
    Context::initialize(argc, argv, sdk_path, gui);
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

bool br_plot_metadata(int num_files, const char *files[], const char *columns, bool show)
{
    return PlotMetadata(QtUtils::toStringList(num_files, files), columns, show);
}

float br_progress()
{
    return Globals->progress();
}

void br_read_line(int *argc, const char ***argv)
{
    static QList<QByteArray> byteArrayList;
    static QVector<const char*> rawCharArrayList;

    byteArrayList.clear(); rawCharArrayList.clear();
    foreach (const QString &string, QtUtils::parse(QTextStream(stdin).readLine(), ' ')) {
        byteArrayList.append(string.toLocal8Bit());
        rawCharArrayList.append(byteArrayList.last().data());
    }

    *argc = byteArrayList.size();
    *argv = rawCharArrayList.data();
}

void br_reformat(const char *target_input, const char *query_input, const char *simmat, const char *output)
{
    Output::reformat(TemplateList::fromGallery(target_input).files(), TemplateList::fromGallery(query_input).files(), simmat, output);
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
