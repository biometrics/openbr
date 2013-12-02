#ifdef BR_LIBRARY
  #define JANUS_LIBRARY
#endif

#include "janus.h"
#include "openbr_plugin.h"

// Use the provided default implementation of some functions
#include "janus/src/janus.cpp"

janus_error janus_initialize(const char *sdk_path, const char *model_file)
{
    int argc = 1;
    const char *argv[1] = { "" };
    br::Context::initialize(argc, (char**)argv, sdk_path);
    QString algorithm = model_file;
    if (algorithm.isEmpty()) algorithm = "FaceRecognition";
    br::Globals->algorithm = algorithm;
    return JANUS_SUCCESS;
}

void janus_finalize()
{
    br::Context::finalize();
}

janus_error janus_initialize_template(janus_partial_template *partial_template)
{
    (void) partial_template;
    return JANUS_SUCCESS;
}

janus_error janus_add_image(const janus_image image, const janus_attribute_list attributes, janus_partial_template partial_template)
{
    (void) image;
    (void) attributes;
    (void) partial_template;
    return JANUS_SUCCESS;
}

janus_error janus_finalize_template(janus_partial_template partial_template, janus_template template_, size_t *bytes)
{
    (void) partial_template;
    (void) template_;
    *bytes = 0;
    return JANUS_SUCCESS;
}

janus_error janus_verify(const janus_template a, const size_t a_bytes, const janus_template b, const size_t b_bytes, float *similarity)
{
    (void) a;
    (void) a_bytes;
    (void) b;
    (void) b_bytes;
    *similarity = 0;
    return JANUS_SUCCESS;
}
