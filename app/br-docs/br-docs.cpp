
#include <QRegularExpression>
#include <stdio.h>

#include <openbr/openbr_plugin.h>

int main(int argc, char *argv[])
{
    QRegularExpression regex;

    for (int i = 1; i < argc; i++) {
        QString arg = QString::fromLocal8Bit(argv[i]);
        if (arg == "--help" || arg == "-h") {
            printf("Usage: br-docs\n");
            printf("  --regex <pattern>   Only generate docs for transforms matching <pattern>\n");
            return 0;
        } else if (arg == "--regex") {
          regex = QRegularExpression(QString::fromLocal8Bit(argv[++i]));
        }
    }

    br::Context::initialize(argc, argv);

    br::AllDocs(regex);

    br::Context::finalize();
}
