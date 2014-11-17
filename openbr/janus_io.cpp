#include <QtConcurrent>

#define JANUS_CUSTOM_ADD_SAMPLE
#define JANUS_CUSTOM_CREATE_GALLERY
#define JANUS_CUSTOM_CREATE_TEMPLATES
#include "janus/src/janus_io.cpp"

static void _janus_add_sample(vector<double> &samples, double sample)
{
    static QMutex sampleLock;
    QMutexLocker sampleLocker(&sampleLock);
    samples.push_back(sample);
}

static void _janus_create_template(const char *data_path, TemplateData templateData, janus_gallery gallery, bool verbose)
{
    janus_template template_;
    janus_template_id templateID;
    JANUS_ASSERT(TemplateIterator::create(data_path, templateData, &template_, &templateID, verbose))

    static QMutex enrollLock;
    QMutexLocker enrollLocker(&enrollLock);

    JANUS_ASSERT(janus_enroll(template_, templateID, gallery))
}

janus_error janus_create_gallery(const char *data_path, janus_metadata metadata, janus_gallery gallery, int verbose)
{
    TemplateIterator ti(metadata, true);
    TemplateData templateData = ti.next();
    QFutureSynchronizer<void> futures;
    while (!templateData.templateIDs.empty()) {
        futures.addFuture(QtConcurrent::run(_janus_create_template, data_path, templateData, gallery, verbose));
        templateData = ti.next();
    }
    futures.waitForFinished();
    return JANUS_SUCCESS;
}

janus_error janus_create_templates(const char *data_path, janus_metadata metadata, const char *gallery_file, int verbose)
{
    TemplateIterator ti(metadata, true);
    TemplateData templateData = ti.next();
    unsigned int num_templates = 1;

    janus_gallery gallery;
    JANUS_ASSERT(janus_allocate_gallery(&gallery))
    QFutureSynchronizer<void> futures;
    while (!templateData.templateIDs.empty()) {
        futures.addFuture(QtConcurrent::run(_janus_create_template, data_path, templateData, gallery, verbose));
        templateData = ti.next();
        num_templates++;
    }
    futures.waitForFinished();
    janus_flat_gallery flat_gallery = new janus_data[num_templates*janus_max_template_size()];
    size_t bytes;
    JANUS_ASSERT(janus_flatten_gallery(gallery, flat_gallery, &bytes))

    std::ofstream file;
    file.open(gallery_file, std::ios::out | std::ios::binary | std::ios::ate);
    file.write((char*)flat_gallery, bytes);
    file.close();
    return JANUS_SUCCESS;
}
