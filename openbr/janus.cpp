#include "janus.h"
#include "janus_io.h"
#include "openbr_plugin.h"

using namespace br;

static QSharedPointer<Transform> transform;
static QSharedPointer<Distance> distance;

size_t janus_max_template_size()
{
    return 33554432; // 32 MB
}

janus_error janus_initialize(const char *sdk_path, const char *model_file)
{
    int argc = 1;
    const char *argv[1] = { "janus" };
    Context::initialize(argc, (char**)argv, sdk_path, false);
    Globals->quiet = true;
    QString algorithm = model_file;
    if (algorithm.isEmpty()) {
        transform = Transform::fromAlgorithm("Cvt(Gray)+Affine(88,88,0.25,0.35)+<FaceRecognitionExtraction>+<FaceRecognitionEmbedding>+<FaceRecognitionQuantization>", false);
        distance = Distance::fromAlgorithm("FaceRecognition");
    } else if (algorithm == "PP5") {
        transform.reset(Transform::make("PP5Enroll", NULL));
        distance.reset(Distance::make("PP5Compare", NULL));
    } else {
        transform = Transform::fromAlgorithm(algorithm, false);
        distance = Distance::fromAlgorithm(algorithm);
    }
    return JANUS_SUCCESS;
}

janus_error janus_finalize()
{
    transform.reset();
    distance.reset();
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
    *bytes = 0;
    foreach (const cv::Mat &m, *template_) {
        assert(m.isContinuous());
        const size_t templateBytes = m.rows * m.cols * m.elemSize();
        if (*bytes + sizeof(size_t) + templateBytes > janus_max_template_size())
            break;
        memcpy(flat_template, &templateBytes, sizeof(templateBytes));
        flat_template += sizeof(templateBytes);
        memcpy(flat_template, m.data, templateBytes);
        flat_template += templateBytes;
        *bytes += sizeof(size_t) + templateBytes;
    }

    delete template_;
    return JANUS_SUCCESS;
}

janus_error janus_verify(const janus_flat_template a, const size_t a_bytes, const janus_flat_template b, const size_t b_bytes, float *similarity)
{
    *similarity = 0;

    int comparisons = 0;
    janus_flat_template a_template = a;
    while (a_template < a + a_bytes) {
        const size_t a_template_bytes = *reinterpret_cast<size_t*>(a_template);
        a_template += sizeof(a_template_bytes);

        janus_flat_template b_template = b;
        while (b_template < b + b_bytes) {
                const size_t b_template_bytes = *reinterpret_cast<size_t*>(b_template);
                b_template += sizeof(b_template_bytes);

                *similarity += distance->compare(cv::Mat(1, a_template_bytes, CV_8UC1, a_template),
                                                 cv::Mat(1, b_template_bytes, CV_8UC1, b_template));
                comparisons++;

                b_template += b_template_bytes;
        }

        a_template += a_template_bytes;
    }

    if (*similarity != *similarity) // True for NaN
        return JANUS_UNKNOWN_ERROR;

    if (comparisons > 0) *similarity /= comparisons;
    else                 *similarity = -std::numeric_limits<float>::max();
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

janus_error janus_gallery_size(janus_gallery gallery, size_t *size)
{
    *size = TemplateList::fromGallery(gallery).size();
    return JANUS_SUCCESS;
}

janus_error janus_compare(janus_gallery target, janus_gallery query, float *similarity_matrix, janus_template_id *target_ids, janus_template_id *query_ids)
{
    const TemplateList targets = TemplateList::fromGallery(target);
    const TemplateList queries = TemplateList::fromGallery(query);
    QScopedPointer<MatrixOutput> matrix(MatrixOutput::make(targets.files(), queries.files()));
    distance->compare(targets, queries, matrix.data());
    const QVector<janus_template_id> targetIds = File::get<janus_template_id,File>(matrix->targetFiles, "TEMPLATE_ID").toVector();
    const QVector<janus_template_id> queryIds  = File::get<janus_template_id,File>(matrix->queryFiles,  "TEMPLATE_ID").toVector();
    memcpy(similarity_matrix, matrix->data.data, matrix->data.rows * matrix->data.cols * sizeof(float));
    memcpy(target_ids, targetIds.data(), targetIds.size() * sizeof(janus_template_id));
    memcpy(query_ids, queryIds.data(), queryIds.size() * sizeof(janus_template_id));
    return JANUS_SUCCESS;
}
