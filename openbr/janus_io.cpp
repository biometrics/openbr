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
    templateData.release();

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

typedef QPair<janus_template_id, FlatTemplate> TemplatePair;

TemplatePair _janus_create_flat_template(const char *data_path, TemplateData templateData, bool verbose)
{
    janus_template template_;
    janus_template_id templateID;
    JANUS_ASSERT(TemplateIterator::create(data_path, templateData, &template_, &templateID, verbose))
    templateData.release();
    return TemplatePair(templateID, FlatTemplate(template_));
}

janus_error janus_create_templates(const char *data_path, janus_metadata metadata, const char *gallery_file, int verbose)
{
    TemplateIterator ti(metadata, true);
    TemplateData templateData = ti.next();

    QFutureSynchronizer<TemplatePair> futures;
    while (!templateData.templateIDs.empty()) {
        futures.addFuture(QtConcurrent::run(_janus_create_flat_template, data_path, templateData, verbose));
        templateData = ti.next();
    }
    futures.waitForFinished();
    QList< QFuture<TemplatePair> > flat_templates = futures.futures();

    std::ofstream file;
    file.open(gallery_file, std::ios::out | std::ios::binary);
    foreach (const QFuture<TemplatePair> &future, flat_templates) {
        janus_template_id templateID = future.result().first;
        FlatTemplate flatTemplate = future.result().second;
        file.write((char*)&templateID, sizeof(templateID));
        file.write((char*)&flatTemplate.data->bytes, sizeof(flatTemplate.data->bytes));
        file.write((char*)flatTemplate.data->flat_template, flatTemplate.data->bytes);
    }

    file.close();
    return JANUS_SUCCESS;
}
