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

#include <QCoreApplication>
#include <QRunnable>
#include <QThreadPool>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openbr/openbr_plugin.h>

/*!
 * \defgroup cli Command Line Interface
 * \brief Command line wrapper of the \ref c_sdk.
 *
 * The easiest and fastest way to run algorithms and evaluating results, we use it all the time!
 * Commands are designed to mirror the \ref c_sdk and are evaluated in the order they are entered.
 * To get started, try running:
 * \code
 * $ br -help
 * \endcode
 *
 * \section cli_examples Examples
 * - \ref cli_show_face_detection
 * - \ref cli_age_estimation
 * - \ref cli_face_recognition
 * - \ref cli_face_recognition_evaluation
 * - \ref cli_gender_estimation
 * - \ref cli_show_face_detection
 */

/*!
 * \ingroup cli
 * \page cli_show_face_detection Show Face Detection
 * \code
 * $ br -algorithm ShowFaceDetection -enrollAll -enroll ../data/family.jpg # Press 'Enter' to cycle through the results
 * \endcode
 */

class FakeMain : public QRunnable
{
    int argc;
    char **argv;

public:
    FakeMain(int argc_, char **argv_)
        : argc(argc_), argv(argv_) {}

    void run()
    {
        // Remove program name
        argv = &argv[1];
        argc--;

        if (argc == 0) printf("%s\nTry running 'br -help'\n", br_about());

        bool daemon = false;
        const char *daemon_pipe = NULL;
        while (daemon || (argc > 0)) {
            const char *fun;
            int parc;
            const char **parv;
            if (argc == 0)
                br_read_pipe(daemon_pipe, &argc, &argv);

            fun = argv[0];
            if (fun[0] == '-') fun++;
            parc = 0;
            bool isNumber = false;
            QString(argv[parc+1]).toDouble(&isNumber);
            while ((parc+1 < argc) && ((argv[parc+1][0] != '-') || isNumber)) {
                parc++;
                QString(argv[parc+1]).toDouble(&isNumber);
            }
            parv = (const char **)&argv[1];
            argc = argc - (parc+1);
            argv = &argv[parc+1];

            // Core Tasks
            if (!strcmp(fun, "train")) {
                check(parc >= 1, "Insufficient parameter count for 'train'.");
                br_train_n(parc == 1 ? 1 : parc-1, parv, parc == 1 ? "" : parv[parc-1]);
            } else if (!strcmp(fun, "enroll")) {
                check(parc >= 1, "Insufficient parameter count for 'enroll'.");
                if (parc == 1) br_enroll(parv[0]);
                else           br_enroll_n(parc-1, parv, parv[parc-1]);
            } else if (!strcmp(fun, "compare")) {
                check((parc >= 2) && (parc <= 3), "Incorrect parameter count for 'compare'.");
                br_compare(parv[0], parv[1], parc == 3 ? parv[2] : "");
            } else if (!strcmp(fun, "eval")) {
                check((parc >= 1) && (parc <= 4), "Incorrect parameter count for 'eval'.");
                if (parc == 1) {
                    br_eval(parv[0], "", "", 0);
                } else if (parc == 2) {
                    if (br::File(parv[1]).suffix() == "csv") {
                        br_eval(parv[0], "", parv[1], 0);
                    } else if (br::File(parv[1]).suffix() == "mask") {
                        br_eval(parv[0], parv[1], "", 0);
                    } else {
                        br_eval(parv[0], "", "", atoi(parv[1]));
                    }
                } else if (parc == 3) {
                    if (br::File(parv[2]).suffix() == "csv") {
                        br_eval(parv[0], parv[1], parv[2], 0);
                    } else if ( br::File(parv[1]).suffix() == "csv") {
                        br_eval(parv[0], "", parv[1], atoi(parv[2]));
                    } else {
                        br_eval(parv[0], parv[1], "", atoi(parv[2]));
                    }
                } else {
                    br_eval(parv[0], parv[1], parv[2], atoi(parv[3]));
                }
            } else if (!strcmp(fun, "plot")) {
                check(parc >= 2, "Incorrect parameter count for 'plot'.");
                br_plot(parc-1, parv, parv[parc-1], true);
            }

            // Secondary Tasks
            else if (!strcmp(fun, "fuse")) {
                check(parc >= 4, "Insufficient parameter count for 'fuse'.");
                br_fuse(parc-3, parv, parv[parc-3], parv[parc-2], parv[parc-1]);
            } else if (!strcmp(fun, "cluster")) {
                check(parc >= 3, "Insufficient parameter count for 'cluster'.");
                br_cluster(parc-2, parv, atof(parv[parc-2]), parv[parc-1]);
            } else if (!strcmp(fun, "makeMask")) {
                check(parc == 3, "Incorrect parameter count for 'makeMask'.");
                br_make_mask(parv[0], parv[1], parv[2]);
            } else if (!strcmp(fun, "makePairwiseMask")) {
                check(parc == 3, "Incorrect parameter count for 'makePairwiseMask'.");
                br_make_pairwise_mask(parv[0], parv[1], parv[2]);
            } else if (!strcmp(fun, "combineMasks")) {
                check(parc >= 4, "Insufficient parameter count for 'combineMasks'.");
                br_combine_masks(parc-2, parv, parv[parc-2], parv[parc-1]);
            } else if (!strcmp(fun, "cat")) {
                check(parc >= 2, "Insufficient parameter count for 'cat'.");
                br_cat(parc-1, parv, parv[parc-1]);
            } else if (!strcmp(fun, "convert")) {
                check(parc == 3, "Incorrect parameter count for 'convert'.");
                br_convert(parv[0], parv[1], parv[2]);
            } else if (!strcmp(fun, "assertEval")) {
                check(parc == 3, "Incorrect parameter count for 'assertEval'.");
                br_assert_eval(parv[0], parv[1], atof(parv[2]));
            } else if (!strcmp(fun, "evalClassification")) {
                check(parc >= 2 && parc <= 4, "Incorrect parameter count for 'evalClassification'.");
                br_eval_classification(parv[0], parv[1], parc >= 3 ? parv[2] : "", parc >= 4 ? parv[3] : "");
            } else if (!strcmp(fun, "evalClustering")) {
                check((parc >= 2) && (parc <= 5), "Incorrect parameter count for 'evalClustering'.");
                br_eval_clustering(parv[0], parv[1], parc > 2 ? parv[2] : "", parc > 3 ? atoi(parv[3]) : 1, parc > 4 ? parv[4] : "");
            } else if (!strcmp(fun, "evalDetection")) {
                check((parc >= 2) && (parc <= 7), "Incorrect parameter count for 'evalDetection'.");
                br_eval_detection(parv[0], parv[1], parc >= 3 ? parv[2] : "", parc >= 4 ? atoi(parv[3]) : 0, parc >= 5 ? atoi(parv[4]) : 0, parc >= 6 ? atoi(parv[5]) : 0, parc >= 7 ? atof(parv[6]) : 0);
            } else if (!strcmp(fun, "evalLandmarking")) {
                check((parc >= 2) && (parc <= 7), "Incorrect parameter count for 'evalLandmarking'.");
                br_eval_landmarking(parv[0], parv[1], parc >= 3 ? parv[2] : "", parc >= 4 ? atoi(parv[3]) : 0, parc >= 5 ? atoi(parv[4]) : 1,  parc >= 6 ? atoi(parv[5]) : 0, parc >= 7 ? atoi(parv[6]) : 5);
            } else if (!strcmp(fun, "evalRegression")) {
                check(parc >= 2 && parc <= 4, "Incorrect parameter count for 'evalRegression'.");
                br_eval_regression(parv[0], parv[1], parc >= 3 ? parv[2] : "", parc >= 4 ? parv[3] : "");
            } else if (!strcmp(fun, "evalKNN")) {
                check(parc >= 2 && parc <= 3, "Incorrect parameter count for 'evalKNN'.");
                br_eval_knn(parv[0], parv[1], parc > 2 ? parv[2] : "");
            } else if (!strcmp(fun, "evalEER")) {
                check(parc >=1 && parc <=4 , "Incorrect parameter count for 'evalEER'.");
                br_eval_eer(parv[0], parc > 1 ? parv[1] : "", parc > 2 ? parv[2] : "", parc > 3 ? parv[3] : "");
            } else if (!strcmp(fun, "pairwiseCompare")) {
                check((parc >= 2) && (parc <= 3), "Incorrect parameter count for 'pairwiseCompare'.");
                br_pairwise_compare(parv[0], parv[1], parc == 3 ? parv[2] : "");
            } else if (!strcmp(fun, "inplaceEval")) {
                check((parc >= 3) && (parc <= 4), "Incorrect parameter count for 'inplaceEval'.");
                br_inplace_eval(parv[0], parv[1], parv[2], parc == 4 ? parv[3] : "");
            } else if (!strcmp(fun, "plotDetection")) {
                check(parc >= 2, "Incorrect parameter count for 'plotDetection'.");
                br_plot_detection(parc-1, parv, parv[parc-1], true);
            } else if (!strcmp(fun, "plotLandmarking")) {
                check(parc >= 2, "Incorrect parameter count for 'plotLandmarking'.");
                br_plot_landmarking(parc-1, parv, parv[parc-1], true);
            } else if (!strcmp(fun, "plotMetadata")) {
                check(parc >= 2, "Incorrect parameter count for 'plotMetadata'.");
                br_plot_metadata(parc-1, parv, parv[parc-1], true);
            } else if (!strcmp(fun, "plotKNN")) {
                check(parc >=2, "Incorrect parameter count for 'plotKNN'.");
                br_plot_knn(parc-1, parv, parv[parc-1], true);
            } else if (!strcmp(fun, "plotEER")) {
                check(parc >= 2, "Incorrect parameter count for 'plotEER'.");
                br_plot_eer(parc-1, parv, parv[parc-1], true);
            } else if (!strcmp(fun, "project")) {
                check(parc == 2, "Insufficient parameter count for 'project'.");
                br_project(parv[0], parv[1]);
            } else if (!strcmp(fun, "deduplicate")) {
                check(parc == 3, "Incorrect parameter count for 'deduplicate'.");
                br_deduplicate(parv[0], parv[1], parv[2]);
            } else if (!strcmp(fun, "likely")) {
                check(parc == 3, "Incorrect parameter count for 'likely'.");
                br_likely(parv[0], parv[1], parv[2]);
            }

            // Miscellaneous
            else if (!strcmp(fun, "help")) {
                check(parc == 0, "No parameters expected for 'help'.");
                help();
            } else if (!strcmp(fun, "gui")) {
                // Do nothing because we checked for this flag prior to initialization
            } else if (!strcmp(fun, "objects")) {
                check(parc <= 2, "Incorrect parameter count for 'objects'.");
                int size = br_objects(NULL, 0, parc >= 1 ? parv[0] : ".*", parc >= 2 ? parv[1] : ".*");
                char *temp = new char[size];
                br_objects(temp, size, parc >= 1 ? parv[0] : ".*", parc >= 2 ? parv[1] : ".*");
                printf("%s\n", temp);
                delete [] temp;
            } else if (!strcmp(fun, "about")) {
                check(parc == 0, "No parameters expected for 'about'.");
                printf("%s\n", br_about());
            } else if (!strcmp(fun, "version")) {
                check(parc == 0, "No parameters expected for 'version'.");
                printf("%s\n", br_version());
            } else if (!strcmp(fun, "daemon")) {
                check(parc == 1, "Incorrect parameter count for 'daemon'.");
                daemon = true;
                daemon_pipe = parv[0];
            } else if (!strcmp(fun, "slave")) {
                // This is used internally by processWrapper, if you want to remove it, also remove
                // plugins/core/processwrapper.cpp
                check(parc == 1, "Incorrect parameter count for 'slave'");
                br_slave_process(parv[0]);
            } else if (!strcmp(fun, "exit")) {
                check(parc == 0, "No parameters expected for 'exit'.");
                daemon = false;
            } else if (!strcmp(fun, "getHeader")) {
                check(parc == 1, "Incorrect parameter count for 'getHeader'.");
                const char *target_gallery, *query_gallery;
                br_get_header(parv[0], &target_gallery, &query_gallery);
                printf("%s\n%s\n", target_gallery, query_gallery);
            } else if (!strcmp(fun, "setHeader")) {
                check(parc == 3, "Incorrect parameter count for 'setHeader'.");
                br_set_header(parv[0], parv[1], parv[2]);
            } else if (!strcmp(fun, "srand")) {
                check(parc == 1, "Incorrect parameter count for 'srand'.");
                srand(atoi(parv[1]));
            } else if (!strcmp(fun, "br")) {
                printf("That's me!\n");
            } else if (parc <= 1) {
                br_set_property(fun, parc >=1 ? parv[0] : "");
            } else {
                printf("Unrecognized function '%s'\n", fun);
            }
        }

        QCoreApplication::exit();
    }

private:
    static void check(bool condition, const char *error_message)
    {
        if (!condition) {
            printf("%s\n", error_message);
            QCoreApplication::exit();
        }
    }

    static void help()
    {
        printf("<arg> = Input; {arg} = Output; [arg] = Optional; (arg0|...|argN) = Choice\n"
               "\n"
               "==== Core Commands ====\n"
               "-train <gallery> ... <gallery> [{model}]\n"
               "-enroll <input_gallery> ... <input_gallery> {output_gallery}\n"
               "-compare <target_gallery> <query_gallery> [{output}]\n"
               "-eval <simmat> [<mask>] [{csv}] [{matches}]\n"
               "-plot <csv> ... <csv> {destination}\n"
               "\n"
               "==== Other Commands ====\n"
               "-fuse <simmat> ... <simmat> (None|MinMax|ZScore|WScore) (Min|Max|Sum[W1:W2:...:Wn]|Replace|Difference|None) {simmat}\n"
               "-cluster <simmat> ... <simmat> <aggressiveness> {csv}\n"
               "-makeMask <target_gallery> <query_gallery> {mask}\n"
               "-makePairwiseMask <target_gallery> <query_gallery> {mask}\n"
               "-combineMasks <mask> ... <mask> {mask} (And|Or)\n"
               "-cat <gallery> ... <gallery> {gallery}\n"
               "-convert (Format|Gallery|Output) <input_file> {output_file}\n"
               "-evalClassification <predicted_gallery> <truth_gallery> <predicted property name> <ground truth proprty name>\n"
               "-evalClustering <clusters> <truth_gallery> [truth_property [cluster_csv [cluster_property]]]\n"
               "-evalDetection <predicted_gallery> <truth_gallery> [{csv}] [{normalize}] [{minSize}] [{maxSize}]\n"
               "-evalLandmarking <predicted_gallery> <truth_gallery> [{csv} [<normalization_index_a> <normalization_index_b>] [sample_index] [total_examples]]\n"
               "-evalRegression <predicted_gallery> <truth_gallery> <predicted property name> <ground truth property name>\n"
               "-evalKNN <knn_graph> <knn_truth> [{csv}]\n"
               "-pairwiseCompare <target_gallery> <query_gallery> [{output}]\n"
               "-inplaceEval <simmat> <target> <query> [{csv}]\n"
               "-assertEval <simmat> <mask> <accuracy>\n"
               "-plotDetection <file> ... <file> {destination}\n"
               "-plotLandmarking <file> ... <file> {destination}\n"
               "-plotMetadata <file> ... <file> <columns>\n"
               "-plotKNN <file> ... <file> {destination}\n"
               "-plotEER <file> ... <file> {destination}\n"
               "-project <input_gallery> {output_gallery}\n"
               "-deduplicate <input_gallery> <output_gallery> <threshold>\n"
               "-likely <input_type> <output_type> <output_likely_source>\n"
               "-getHeader <matrix>\n"
               "-setHeader {<matrix>} <target_gallery> <query_gallery>\n"
               "-<key> <value>\n"
               "\n"
               "==== Miscellaneous ====\n"
               "-help\n"
               "-gui\n"
               "-objects [abstraction [implementation]]\n"
               "-about\n"
               "-version\n"
               "-daemon\n"
               "-slave\n"
               "-exit\n"
               "-srand <int>\n");
    }
};

int main(int argc, char *argv[])
{
    const bool gui         = (argc >= 2) && !strcmp(argv[1], "-gui");
    const bool noEventLoop = (argc >= 2) && !strcmp(argv[1], "-noEventLoop");
    br_initialize(argc, argv, "", gui);

    if (noEventLoop) {
        FakeMain(argc, argv).run();
    } else {
        QThreadPool::globalInstance()->start(new FakeMain(argc, argv));
        QCoreApplication::exec();
    }

    br_finalize();
}
