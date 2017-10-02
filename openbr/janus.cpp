#include "iarpa_janus.h"
#include "iarpa_janus_io.h"
#include "openbr_plugin.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/common.h"
#include <fstream>
using namespace br;

static QSharedPointer<Transform> detect;
static QSharedPointer<Transform> augment;
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
    Globals->enrollAll = true;
    Globals->file.set(QString("temp_path"), QString(temp_path));
    const QString algorithm = model_file;
    detect.reset(Transform::make("Cvt(Gray)+Cascade(FrontalFace,ROCMode=true)", NULL));
    if (algorithm.isEmpty()) {
        augment.reset(Transform::make("Cvt(Gray)+Affine(88,88,0.25,0.35)+<FaceRecognitionExtraction>+<FaceRecognitionEmbedding>+<FaceRecognitionQuantization>", NULL));
        distance = Distance::fromAlgorithm("FaceRecognition");
    } else {
        augment = Transform::fromAlgorithm(algorithm);
        distance = Distance::fromAlgorithm(algorithm);
    }
    return JANUS_SUCCESS;
}

janus_error janus_finalize()
{
    detect.reset();
    augment.reset();
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

bool compareConfidence(Template a, Template b) { return (a.file.get<float>("Confidence") > b.file.get<float>("Confidence")); }

janus_error janus_detect(const janus_image image, janus_attributes *attributes_array, const size_t num_requested, size_t *num_actual)
{
    TemplateList src, dst;

    Template t;
    cv::Mat input(image.height,
                  image.width,
                  image.color_space == JANUS_GRAY8 ? CV_8UC1 : CV_8UC3,
                  image.data);

    t.append(input);
    src.append(t);
    detect->project(src, dst);
    *num_actual = dst.size();
    if (dst.size() == 0)
        return JANUS_FAILURE_TO_DETECT;

    // Sort by confidence, descending
    std::sort(dst.begin(), dst.end(), compareConfidence);

    size_t count = 0;
    foreach (const Template &temp, dst) {
        QRectF rect = temp.file.rects().first();
        attributes_array->face_x = rect.x();
        attributes_array->face_y = rect.y();
        attributes_array->face_width = rect.width();
        attributes_array->face_height = rect.height();
        attributes_array->detection_confidence = temp.file.get<float>("Confidence");
        attributes_array++;
        if (++count >= num_requested)
            break;
    }
    attributes_array -= count;
    return JANUS_SUCCESS;
}

janus_error janus_augment(const janus_image image, janus_attributes *attributes, janus_template template_)
{
    Template t;
    if (std::isnan(attributes->face_x) ||
        std::isnan(attributes->face_y) ||
        std::isnan(attributes->face_width) ||
        std::isnan(attributes->face_height))
        return JANUS_MISSING_ATTRIBUTES;

    QRectF rect(attributes->face_x,
                attributes->face_y,
                attributes->face_width,
                attributes->face_height);

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
    if (!std::isnan(attributes->right_eye_x) &&
        !std::isnan(attributes->right_eye_y) &&
        !std::isnan(attributes->left_eye_x) &&
        !std::isnan(attributes->left_eye_y)) {
        t.file.set("Affine_0", QPointF(attributes->right_eye_x - rect.x(), attributes->right_eye_y - rect.y()));
        t.file.set("Affine_1", QPointF(attributes->left_eye_x - rect.x(), attributes->left_eye_y - rect.y()));
        t.file.set("First_Eye", t.file.get<QPointF>("Affine_0"));
        t.file.set("Second_Eye", t.file.get<QPointF>("Affine_1"));
        t.file.appendPoint(t.file.get<QPointF>("Affine_0"));
        t.file.appendPoint(t.file.get<QPointF>("Affine_1"));
    }
    Template u;
    augment->project(t, u);
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

void unflatten_template(const janus_flat_template flat_template, const size_t template_bytes, janus_gallery gallery, const janus_template_id template_id)
{
    janus_template t;
    JANUS_ASSERT(janus_allocate_template(&t))
    t->file.set("TEMPLATE_ID", QString::number((int)template_id));
    janus_flat_template flat_template_ = flat_template;

    while (flat_template_ < flat_template + template_bytes) {
        size_t bytes = *reinterpret_cast<size_t*>(flat_template_);
        flat_template_ += sizeof(bytes);

        t->append(cv::Mat(1, bytes, CV_8UC1, flat_template_).clone());
        flat_template_ += bytes;
    }
    gallery->append(t);
}

janus_error janus_write_gallery(const janus_flat_template *templates, const size_t *templates_bytes, const janus_template_id *template_ids, const size_t num_templates, janus_gallery_path gallery_path)
{
    std::ofstream file;
    file.open(gallery_path, std::ios::out | std::ios::binary);

    for (size_t i=0; i<num_templates; i++) {
        file.write((char*)&template_ids[i], sizeof(janus_template_id));
        file.write((char*)&templates_bytes[i], sizeof(size_t));
        file.write((char*)templates[i], templates_bytes[i]);
    }

    file.close();
    return JANUS_SUCCESS;
}

janus_error janus_open_gallery(janus_gallery_path gallery_path, janus_gallery *gallery)
{
    *gallery = new janus_gallery_type();
    std::ifstream file;
    file.open(gallery_path, std::ios::in | std::ios::binary | std::ios::ate);
    const size_t bytes = file.tellg();
    file.seekg(0, std::ios::beg);
    janus_data *templates = new janus_data[bytes];
    file.read((char*)templates, bytes);
    file.close();

    janus_data *templates_ = templates;
    while (templates_ < templates + bytes) {
        janus_template_id template_id = *reinterpret_cast<janus_template_id*>(templates_);
        templates_ += sizeof(janus_template_id);
        const size_t template_bytes = *reinterpret_cast<size_t*>(templates_);
        templates_ += sizeof(size_t);

        janus_flat_template flat_template = new janus_data[template_bytes];
        memcpy(flat_template, templates_, template_bytes);
        templates_ += template_bytes;

        unflatten_template(flat_template, template_bytes, *gallery, template_id);
        delete[] flat_template;
    }

    delete[] templates;
    return JANUS_SUCCESS;
}

janus_error janus_close_gallery(janus_gallery gallery)
{
    delete gallery;
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

janus_error janus_search(const janus_flat_template probe, const size_t probe_bytes, const janus_gallery gallery, const size_t requested_returns, janus_template_id *template_ids, float *similarities, size_t *actual_returns)
{
    typedef QPair<float, int> Pair;
    QList<Pair> comparisons; comparisons.reserve(requested_returns);
    foreach (const janus_template &target_template, *gallery) {
        janus_template_id target_id = target_template->file.get<janus_template_id>("TEMPLATE_ID");

        size_t target_bytes;
        janus_data *buffer = new janus_data[janus_max_template_size()];
        JANUS_ASSERT(janus_flatten_template(target_template, buffer, &target_bytes))

        janus_flat_template target_flat = new janus_data[target_bytes];
        memcpy(target_flat, buffer, target_bytes);

        float similarity;
        JANUS_ASSERT(janus_verify(probe, probe_bytes, target_flat, target_bytes, &similarity))
        if ((size_t)comparisons.size() < requested_returns) {
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
        delete[] buffer;
        delete[] target_flat;
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
