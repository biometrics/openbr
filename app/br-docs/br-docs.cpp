
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <openbr/openbr_plugin.h>

int main(int argc, char *argv[])
{
    br::Context::initialize(argc, argv);

    br::AllDocs();

    br::Context::finalize();
}
