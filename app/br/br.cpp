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
#include <openbr/openbr.h>

/*!
 * \defgroup cli Command Line Interface
 * \brief Command line application for running algorithms and evaluating results.
 *
 * The easiest and fastest way to leverage the project, we use it all the time!
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
            parc = 0; while ((parc+1 < argc) && (argv[parc+1][0] != '-')) parc++;
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
                check((parc >= 2) && (parc <= 3), "Incorrect parameter count for 'eval'.");
                br_eval(parv[0], parv[1], parc == 3 ? parv[2] : "");
            } else if (!strcmp(fun, "plot")) {
                check(parc >= 2, "Incorrect parameter count for 'plot'.");
                br_plot(parc-1, parv, parv[parc-1], true);
            }

            // Secondary Tasks
            else if (!strcmp(fun, "fuse")) {
                check(parc >= 5, "Insufficient parameter count for 'fuse'.");
                br_fuse(parc-4, parv, parv[parc-4], parv[parc-3], parv[parc-2], parv[parc-1]);
            } else if (!strcmp(fun, "cluster")) {
                check(parc >= 3, "Insufficient parameter count for 'cluster'.");
                br_cluster(parc-2, parv, atof(parv[parc-2]), parv[parc-1]);
            } else if (!strcmp(fun, "makeMask")) {
                check(parc == 3, "Incorrect parameter count for 'makeMask'.");
                br_make_mask(parv[0], parv[1], parv[2]);
            } else if (!strcmp(fun, "combineMasks")) {
                check(parc >= 4, "Insufficient parameter count for 'combineMasks'.");
                br_combine_masks(parc-2, parv, parv[parc-2], parv[parc-1]);
            } else if (!strcmp(fun, "cat")) {
                check(parc >= 2, "Insufficient parameter count for 'cat'.");
                br_cat(parc-1, parv, parv[parc-1]);
            } else if (!strcmp(fun, "convert")) {
                check(parc == 2, "Incorrect parameter count for 'convert'.");
                br_convert(parv[0], parv[1]);
            } else if (!strcmp(fun, "reformat")) {
                check(parc == 4, "Incorrect parameter count for 'reformat'.");
                br_reformat(parv[0], parv[1], parv[2], parv[3]);
            } else if (!strcmp(fun, "evalClassification")) {
                check(parc == 2, "Incorrect parameter count for 'evalClassification'.");
                br_eval_classification(parv[0], parv[1]);
            } else if (!strcmp(fun, "evalRegression")) {
                check(parc == 2, "Incorrect parameter count for 'evalRegression'.");
                br_eval_regression(parv[0], parv[1]);
            } else if (!strcmp(fun, "evalClusters")) {
                check(parc == 2, "Incorrect parameter count for 'evalClusters'.");
                br_eval_clustering(parv[0], parv[1]);
            } else if (!strcmp(fun, "confusion")) {
                check(parc == 2, "Incorrect parameter count for 'confusion'.");
                int true_positives, false_positives, true_negatives, false_negatives;
                br_confusion(parv[0], atof(parv[1]),
                        &true_positives, &false_positives, &true_negatives, &false_negatives);
                printf("True Positives = %d\nFalse Positives = %d\nTrue Negatives = %d\nFalseNegatives = %d\n",
                       true_positives, false_positives, true_negatives, false_negatives);
            } else if (!strcmp(fun, "plotMetadata")) {
                check(parc >= 2, "Incorrect parameter count for 'plotMetadata'.");
                br_plot_metadata(parc-1, parv, parv[parc-1], true);
            }

            // Miscellaneous
            else if (!strcmp(fun, "help")) {
                check(parc == 0, "No parameters expected for 'help'.");
                help();
            } else if (!strcmp(fun, "objects")) {
                check(parc <= 2, "Incorrect parameter count for 'objects'.");
                printf("%s\n", br_objects(parc >= 1 ? parv[0] : ".*", parc >= 2 ? parv[1] : ".*"));
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
            } else if (!strcmp(fun, "exit")) {
                check(parc == 0, "No parameters expected for 'exit'.");
                daemon = false;
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
               "-eval <simmat> <mask> [{csv}]\n"
               "-plot <file> ... <file> {destination}\n"
               "\n"
               "==== Other Commands ====\n"
               "-fuse <simmat> ... <simmat> <mask> (None|MinMax|ZScore|WScore) (Min|Max|Sum[W1:W2:...:Wn]|Replace|Difference|None) {simmat}\n"
               "-cluster <simmat> ... <simmat> <aggressiveness> {csv}\n"
               "-makeMask <target_gallery> <query_gallery> {mask}\n"
               "-combineMasks <mask> ... <mask> {mask} (And|Or)\n"
               "-cat <gallery> ... <gallery> {gallery}\n"
               "-convert <template> {template}\n"
               "-reformat <target_sigset> <query_sigset> <simmat> {output}\n"
               "-evalClassification <predicted_gallery> <truth_gallery>\n"
               "-evalRegression <predicted_gallery> <truth_gallery>\n"
               "-evalClusters <clusters> <sigset>\n"
               "-confusion <file> <score>\n"
               "-plotMetadata <file> ... <file> <columns>\n"
               "\n"
               "==== Configuration ====\n"
               "-<key> <value>\n"
               "\n"
               "==== Miscellaneous ====\n"
               "-objects [abstraction [implementation]]\n"
               "-about\n"
               "-version\n"
               "-shell\n"
               "-exit\n");
    }
};

int main(int argc, char *argv[])
{
    br_initialize(argc, argv);

    FakeMain *fakeMain = new FakeMain(argc, argv);
    QThreadPool::globalInstance()->start(fakeMain);
    QCoreApplication::exec();

    br_finalize();
}
