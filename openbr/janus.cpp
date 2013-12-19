#ifdef BR_LIBRARY
  #define JANUS_LIBRARY
#endif

#include "janus.h"
#include "openbr_plugin.h"

// Use the provided default implementation of some functions
#include "janus/src/janus.cpp"

using namespace br;

static QSharedPointer<Transform> transform;
static QSharedPointer<Distance> distance;

janus_error janus_initialize(const char *sdk_path, const char *model_file)
{
    int argc = 1;
    const char *argv[1] = { "janus" };
    Context::initialize(argc, (char**)argv, sdk_path);
    QString algorithm = model_file;
    if (algorithm.isEmpty()) algorithm = "Cvt(Gray)+Affine(88,88,0.25,0.35)+<FaceRecognitionExtraction>+<FaceRecognitionEmbedding>+<FaceRecognitionQuantization>:ByteL1";
    transform = Transform::fromAlgorithm(algorithm, false);
    distance = Distance::fromAlgorithm(algorithm);
    return JANUS_SUCCESS;
}

janus_error janus_finalize()
{
    Context::finalize();
    return JANUS_SUCCESS;
}

struct janus_incomplete_template_type
{
    QList<cv::Mat> data;
};

janus_error janus_initialize_template(janus_incomplete_template *incomplete_template)
{
    *incomplete_template = new janus_incomplete_template_type();
    return JANUS_SUCCESS;
}

janus_error janus_add_image(const janus_image image, const janus_attribute_list attributes, janus_incomplete_template incomplete_template)
{
    Template t;
    t.append(cv::Mat(image.height,
                     image.width,
                     image.color_space == JANUS_GRAY8 ? CV_8UC1 : CV_8UC1,
                     image.data));
    for (size_t i=0; i<attributes.size; i++)
        t.file.set(janus_attribute_to_string(attributes.attributes[i]), attributes.values[i]);

    if (!t.file.contains("JANUS_RIGHT_EYE_X") ||
        !t.file.contains("JANUS_RIGHT_EYE_Y") ||
        !t.file.contains("JANUS_LEFT_EYE_X") ||
        !t.file.contains("JANUS_LEFT_EYE_Y"))
        return JANUS_SUCCESS;

    t.file.set("Affine_0", QPointF(t.file.get<float>("JANUS_RIGHT_EYE_X"), t.file.get<float>("JANUS_RIGHT_EYE_Y")));
    t.file.set("Affine_1", QPointF(t.file.get<float>("JANUS_LEFT_EYE_X"), t.file.get<float>("JANUS_LEFT_EYE_Y")));
    Template u;
    transform->project(t, u);
    incomplete_template->data.append(u);
    return JANUS_SUCCESS;
}

janus_error janus_finalize_template(janus_incomplete_template incomplete_template, janus_template template_, size_t *bytes)
{    
    size_t templateBytes = 0;
    size_t numTemplates = 0;
    *bytes = sizeof(templateBytes) + sizeof(numTemplates);
    janus_template pos = template_ + *bytes;

    foreach (const cv::Mat &m, incomplete_template->data) {
        assert(m.isContinuous());
        const size_t currentTemplateBytes = m.rows * m.cols * m.elemSize();
        if (templateBytes == 0)
            templateBytes = currentTemplateBytes;
        if (templateBytes != currentTemplateBytes)
            return JANUS_UNKNOWN_ERROR;
        if (*bytes + templateBytes > JANUS_MAX_TEMPLATE_SIZE)
            break;
        memcpy(pos, m.data, templateBytes);
        *bytes += templateBytes;
        pos = pos + templateBytes;
        numTemplates++;
    }

    *(reinterpret_cast<size_t*>(template_)+0) = templateBytes;
    *(reinterpret_cast<size_t*>(template_)+1) = numTemplates;
    delete incomplete_template;
    return JANUS_SUCCESS;
}

janus_error janus_verify(const janus_template a, const janus_template b, float *similarity)
{
    size_t a_bytes, a_templates, b_bytes, b_templates;
    a_bytes     = *(reinterpret_cast<size_t*>(a)+0);
    a_templates = *(reinterpret_cast<size_t*>(a)+1);
    b_bytes     = *(reinterpret_cast<size_t*>(b)+0);
    b_templates = *(reinterpret_cast<size_t*>(b)+1);
    if (a_bytes != b_bytes)
        return JANUS_UNKNOWN_ERROR;

    float dist = 0;
    for (size_t i=0; i<a_templates; i++)
        for (size_t j=0; j<b_templates; j++)
            dist += distance->compare(cv::Mat(1, a_bytes, CV_8UC1, a+2*sizeof(size_t)+i*a_bytes),
                                      cv::Mat(1, b_bytes, CV_8UC1, b+2*sizeof(size_t)+i*b_bytes));
    *similarity = a_templates * b_templates / dist;
    return JANUS_SUCCESS;
}

struct janus_incomplete_gallery_type
{
    QList< QPair<janus_template, janus_template_id> > templates;
};

janus_error janus_initialize_gallery(janus_incomplete_gallery *incomplete_gallery)
{
    *incomplete_gallery = new janus_incomplete_gallery_type();
    return JANUS_SUCCESS;
}
