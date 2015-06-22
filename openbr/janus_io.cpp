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

typedef QPair<janus_template_id, FlatTemplate> TemplatePair;

TemplatePair _janus_create_flat_template(const char *data_path, TemplateData templateData, bool verbose)
{
    janus_template template_;
    janus_template_id templateID;
    JANUS_ASSERT(TemplateIterator::create(data_path, templateData, &template_, &templateID, verbose))
    templateData.release();
    return TemplatePair(templateID, FlatTemplate(template_));
}

janus_error janus_create_gallery(const char *data_path, janus_metadata metadata, janus_gallery_path gallery_path, int verbose)
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

    vector<janus_flat_template> templates;
    vector<size_t> template_bytes;
    vector<janus_template_id> template_ids;
    size_t num_templates = 0;

    foreach (const QFuture<TemplatePair> &future, flat_templates) {
        template_ids.push_back(future.result().first);
        template_bytes.push_back(future.result().second.data->bytes);
        templates.push_back(future.result().second.data->flat_template);
        num_templates++;
    }

    JANUS_ASSERT(janus_write_gallery(&templates[0], &template_bytes[0], &template_ids[0], num_templates, gallery_path))
    return JANUS_SUCCESS;
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
