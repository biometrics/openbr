#ifdef BR_LIBRARY
  #define JANUS_LIBRARY
#endif

#include "janus.h"
#include "openbr_plugin.h"

janus_error janus_initialize(const char *sdk_path, const char *model_file)
{
    int argc = 1;
    const char *argv[1] = { "br" };
    br::Context::initialize(argc, (char **)argv, sdk_path);
    QString algorithm = model_file;
    if (algorithm.isEmpty()) algorithm = "FaceRecognition";
    br::Globals->setProperty("Algorithm", algorithm);
    return JANUS_SUCCESS;
}

void janus_finalize()
{
    br::Context::finalize();
}