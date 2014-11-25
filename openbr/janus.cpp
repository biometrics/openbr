#include "iarpa_janus.h"
#include "iarpa_janus_io.h"
#include "openbr_plugin.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/common.h"
using namespace br;

static QSharedPointer<Transform> transform;
static QSharedPointer<Distance> distance;

size_t janus_max_template_size()
{
    return 102400;// 100 KB
}

janus_error janus_initialize(const char *sdk_path, const char *temp_path, const char *model_file)
{
    int argc = 1;
    const char *argv[1] = { "janus" };
    Context::initialize(argc, (char**)argv, sdk_path, false);
    Globals->quiet = true;
    Globals->file.set(QString("temp_path"), QString(temp_path));
    const QString algorithm = model_file;
    if (algorithm.isEmpty()) {
        transform.reset(Transform::make("Cvt(Gray)+Affine(88,88,0.25,0.35)+<FaceRecognitionExtraction>+<FaceRecognitionEmbedding>+<FaceRecognitionQuantization>", NULL));
        distance = Distance::fromAlgorithm("FaceRecognition");
    } else if (algorithm.compare("Component") == 0) {
        transform.reset(Transform::make("LandmarksAffine+Cvt(Gray)+<ComponentEnroll>", NULL));
        distance = Distance::fromAlgorithm(algorithm);
     } else {
        transform.reset(Transform::make(algorithm + "Enroll", NULL));
        distance.reset(Distance::make(algorithm + "Compare", NULL));
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

janus_error janus_allocate_template(janus_template *template_)
{
    *template_ = new janus_template_type();
    return JANUS_SUCCESS;
}

janus_error janus_augment(const janus_image image, const janus_attribute_list attributes, janus_template template_)
{
    Template t;
    for (size_t i=0; i<attributes.size; i++)
        t.file.set(janus_attribute_to_string(attributes.attributes[i]), attributes.values[i]);

    if (!t.file.contains("FACE_X") ||
        !t.file.contains("FACE_Y") ||
        !t.file.contains("FACE_WIDTH") ||
        !t.file.contains("FACE_HEIGHT"))
        return JANUS_MISSING_ATTRIBUTES;

    QRectF rect(t.file.get<float>("FACE_X"),
                t.file.get<float>("FACE_Y"),
                t.file.get<float>("FACE_WIDTH"),
                t.file.get<float>("FACE_HEIGHT"));

    if (rect.x() < 0) rect.setX(0);
    if (rect.y() < 0) rect.setY(0);
    if (rect.x() + rect.width() > image.width) rect.setWidth(image.width - rect.x());
    if (rect.y() + rect.height() > image.height) rect.setHeight(image.height - rect.y());

    cv::Mat input(image.height,
                  image.width,
                  image.color_space == JANUS_GRAY8 ? CV_8UC1 : CV_8UC3,
                  image.data);

    input = input(cv::Rect(rect.x(), rect.y(), rect.width(), rect.height())).clone();
    t.append(input);
    if (t.file.contains("RIGHT_EYE_X") &&
        t.file.contains("RIGHT_EYE_Y") &&
        t.file.contains("LEFT_EYE_X") &&
        t.file.contains("LEFT_EYE_Y")) {
        t.file.set("Affine_0", QPointF(t.file.get<float>("RIGHT_EYE_X") - rect.x(), t.file.get<float>("RIGHT_EYE_Y") - rect.y()));
        t.file.set("Affine_1", QPointF(t.file.get<float>("LEFT_EYE_X") - rect.x(), t.file.get<float>("LEFT_EYE_Y") - rect.y()));
        t.file.appendPoint(t.file.get<QPointF>("Affine_0"));
        t.file.appendPoint(t.file.get<QPointF>("Affine_1"));
    }
    Template u;
    transform->project(t, u);
    if (u.file.fte) u.m() = cv::Mat();
    template_->append(u);
    return (u.isEmpty() || !u.first().data) ? JANUS_FAILURE_TO_ENROLL : JANUS_SUCCESS;
}

janus_error janus_flatten_template(janus_template template_, janus_flat_template flat_template, size_t *bytes)
{    
    *bytes = 0;
    foreach (const cv::Mat &m, *template_) {
        if (!m.data)
            continue;

        if (!m.isContinuous())
            return JANUS_UNKNOWN_ERROR;

        const size_t templateBytes = m.rows * m.cols * m.elemSize();
        if (*bytes + sizeof(size_t) + templateBytes > janus_max_template_size())
            break;

        memcpy(flat_template, &templateBytes, sizeof(templateBytes));
        flat_template += sizeof(templateBytes);
        *bytes += sizeof(templateBytes);

        memcpy(flat_template, m.data, templateBytes);
        flat_template += templateBytes;
        *bytes += templateBytes;
    }
    return JANUS_SUCCESS;
}

janus_error janus_free_template(janus_template template_)
{
    delete template_;
    return JANUS_SUCCESS;
}

struct janus_gallery_type : public QList<janus_template>
{};

janus_error janus_allocate_gallery(janus_gallery *gallery_)
{
    *gallery_ = new janus_gallery_type();
    return JANUS_SUCCESS;
}

janus_error janus_enroll(const janus_template template_, const janus_template_id template_id, janus_gallery gallery)
{    
    template_->file.set("TEMPLATE_ID", template_id);
    gallery->push_back(template_);
    return JANUS_SUCCESS;
}

janus_error janus_free_gallery(janus_gallery gallery_) {
    delete gallery_;
    return JANUS_SUCCESS;
}

janus_error janus_flatten_gallery(janus_gallery gallery, janus_flat_gallery flat_gallery, size_t *bytes)
{
    *bytes = 0;
    foreach (const janus_template &t, *gallery) {
        janus_template_id template_id = t->file.get<janus_template_id>("TEMPLATE_ID");

        janus_flat_template u = new janus_data[janus_max_template_size()];
        size_t t_bytes = 0;
        JANUS_ASSERT(janus_flatten_template(t, u, &t_bytes))
        memcpy(flat_gallery, &template_id, sizeof(template_id));
        flat_gallery += sizeof(template_id);
        *bytes += sizeof(template_id);

        memcpy(flat_gallery, &t_bytes, sizeof(t_bytes));
        flat_gallery += sizeof(t_bytes);
        *bytes += sizeof(t_bytes);

        memcpy(flat_gallery, u, t_bytes);
        flat_gallery += t_bytes;
        *bytes += t_bytes;
        delete[] u;
    }
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

janus_error janus_search(const janus_flat_template probe, const size_t probe_bytes, const janus_flat_gallery gallery, const size_t gallery_bytes, int requested_returns, janus_template_id *template_ids, float *similarities, int *actual_returns)
{
    typedef QPair<float, int> Pair;
    QList<Pair> comparisons; comparisons.reserve(requested_returns);
    janus_flat_gallery target_gallery = gallery;
    while (target_gallery < gallery + gallery_bytes) {
        janus_template_id target_id = *reinterpret_cast<janus_template_id*>(target_gallery);
        target_gallery += sizeof(target_id);

        const size_t target_template_bytes = *reinterpret_cast<size_t*>(target_gallery);
        target_gallery += sizeof(target_template_bytes);
        janus_flat_template target_template_flat = new janus_data[target_template_bytes];
        memcpy(target_template_flat, target_gallery, target_template_bytes);
        target_gallery += target_template_bytes;

        float similarity;
        JANUS_ASSERT(janus_verify(probe, probe_bytes, target_template_flat, target_template_bytes, &similarity))
        if (comparisons.size() < requested_returns) {
            comparisons.append(Pair(similarity, target_id));
            std::sort(comparisons.begin(), comparisons.end());
        } else {
            Pair temp = comparisons.first();
            if (temp.first < similarity) {
                comparisons.removeFirst();
                comparisons.append(Pair(similarity, target_id));
                std::sort(comparisons.begin(), comparisons.end());
            }
        }
        delete[] target_template_flat;
    }
    *actual_returns = comparisons.size();
    QList<Pair> temp; temp.reserve(comparisons.size());
    std::reverse_copy(comparisons.begin(), comparisons.end(), std::back_inserter(temp));
    comparisons = temp;
    foreach(const Pair &comparison, comparisons) {
        *similarities = comparison.first; similarities++;
        *template_ids = comparison.second; template_ids++;
    }
    return JANUS_SUCCESS;
}
