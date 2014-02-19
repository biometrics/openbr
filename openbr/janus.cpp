#ifdef BR_LIBRARY
  #define JANUS_LIBRARY
#endif

#include "janus.h"
#include "janus_io.h"
#include "openbr_plugin.h"

using namespace br;

static QSharedPointer<Transform> transform;
static QSharedPointer<Distance> distance;

size_t janus_max_template_size()
{
    return JANUS_MAX_TEMPLATE_SIZE_LIMIT;
}

janus_error janus_initialize(const char *sdk_path, const char *model_file)
{
    int argc = 1;
    const char *argv[1] = { "janus" };
    Context::initialize(argc, (char**)argv, sdk_path);
    QString algorithm = model_file;
    if (algorithm.isEmpty()) {
        transform = Transform::fromAlgorithm("Cvt(Gray)+Affine(88,88,0.25,0.35)+<FaceRecognitionExtraction>+<FaceRecognitionEmbedding>+<FaceRecognitionQuantization>", false);
        distance = Distance::fromAlgorithm("FaceRecognition");
    } else {
        transform = Transform::fromAlgorithm(algorithm, false);
        distance = Distance::fromAlgorithm(algorithm);
    }
    return JANUS_SUCCESS;
}

janus_error janus_finalize()
{
    Context::finalize();
    return JANUS_SUCCESS;
}

struct janus_template_type : public Template
{};

janus_error janus_initialize_template(janus_template *template_)
{
    *template_ = new janus_template_type();
    return JANUS_SUCCESS;
}

janus_error janus_augment(const janus_image image, const janus_attribute_list attributes, janus_template template_)
{
    Template t;
    t.append(cv::Mat(image.height,
                     image.width,
                     image.color_space == JANUS_GRAY8 ? CV_8UC1 : CV_8UC3,
                     image.data));
    for (size_t i=0; i<attributes.size; i++)
        t.file.set(janus_attribute_to_string(attributes.attributes[i]), attributes.values[i]);

    if (!t.file.contains("RIGHT_EYE_X") ||
        !t.file.contains("RIGHT_EYE_Y") ||
        !t.file.contains("LEFT_EYE_X") ||
        !t.file.contains("LEFT_EYE_Y"))
        return JANUS_SUCCESS;

    t.file.set("Affine_0", QPointF(t.file.get<float>("RIGHT_EYE_X"), t.file.get<float>("RIGHT_EYE_Y")));
    t.file.set("Affine_1", QPointF(t.file.get<float>("LEFT_EYE_X"), t.file.get<float>("LEFT_EYE_Y")));
    Template u;
    transform->project(t, u);
    template_->append(u);
    return JANUS_SUCCESS;
}

janus_error janus_finalize_template(janus_template template_, janus_flat_template flat_template, size_t *bytes)
{    
    size_t templateBytes = 0;
    size_t numTemplates = 0;
    *bytes = sizeof(templateBytes) + sizeof(numTemplates);
    janus_flat_template pos = flat_template + *bytes;

    foreach (const cv::Mat &m, *template_) {
        assert(m.isContinuous());
        const size_t currentTemplateBytes = m.rows * m.cols * m.elemSize();
        if (templateBytes == 0)
            templateBytes = currentTemplateBytes;
        if (templateBytes != currentTemplateBytes)
            return JANUS_UNKNOWN_ERROR;
        if (*bytes + templateBytes > janus_max_template_size())
            break;
        memcpy(pos, m.data, templateBytes);
        *bytes += templateBytes;
        pos = pos + templateBytes;
        numTemplates++;
    }

    *(reinterpret_cast<size_t*>(flat_template)+0) = templateBytes;
    *(reinterpret_cast<size_t*>(flat_template)+1) = numTemplates;
    delete template_;
    return JANUS_SUCCESS;
}

janus_error janus_verify(const janus_flat_template a, const size_t a_bytes, const janus_flat_template b, const size_t b_bytes, double *similarity)
{
    (void) a_bytes;
    (void) b_bytes;

    size_t a_template_bytes, a_templates, b_template_bytes, b_templates;
    a_template_bytes = *(reinterpret_cast<size_t*>(a)+0);
    a_templates = *(reinterpret_cast<size_t*>(a)+1);
    b_template_bytes = *(reinterpret_cast<size_t*>(b)+0);
    b_templates = *(reinterpret_cast<size_t*>(b)+1);

    *similarity = 0;
    if ((a_templates == 0) || (b_templates == 0))
        return JANUS_SUCCESS;

    if (a_template_bytes != b_template_bytes)
        return JANUS_UNKNOWN_ERROR;

    for (size_t i=0; i<a_templates; i++)
        for (size_t j=0; j<b_templates; j++)
            *similarity += distance->compare(cv::Mat(1, a_template_bytes, CV_8UC1, a+2*sizeof(size_t)+i*a_template_bytes),
                                             cv::Mat(1, b_template_bytes, CV_8UC1, b+2*sizeof(size_t)+j*b_template_bytes));

    if (*similarity != *similarity) // True for NaN
        return JANUS_UNKNOWN_ERROR;

    *similarity /= a_templates * b_templates;
    return JANUS_SUCCESS;
}

janus_error janus_enroll(const janus_template template_, const janus_template_id template_id, janus_gallery gallery)
{
    template_->file.set("TEMPLATE_ID", template_id);
    QFile file(gallery);
    if (!file.open(QFile::WriteOnly | QFile::Append))
        return JANUS_WRITE_ERROR;
    QDataStream stream(&file);
    stream << *template_;
    file.close();
    delete template_;
    return JANUS_SUCCESS;
}
