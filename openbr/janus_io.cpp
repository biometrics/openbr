#include <QtConcurrent>

#define JANUS_CUSTOM_ADD_SAMPLE
#define JANUS_CUSTOM_CREATE_GALLERY
#include "janus/src/janus_io.cpp"

static void _janus_add_sample(vector<double> &samples, double sample)
{
    static QMutex sampleLock;
    QMutexLocker sampleLocker(&sampleLock);
    samples.push_back(sample);
}

static void _janus_create_template(const char *data_path, TemplateData templateData, janus_gallery gallery)
{
    janus_template template_;
    janus_template_id templateID;
    JANUS_ASSERT(TemplateIterator::create(data_path, templateData, &template_, &templateID))

    static QMutex enrollLock;
    QMutexLocker enrollLocker(&enrollLock);

    JANUS_ASSERT(janus_enroll(template_, templateID, gallery))
}

janus_error janus_create_gallery(const char *data_path, janus_metadata metadata, janus_gallery gallery)
{
    TemplateIterator ti(metadata, true);
    TemplateData templateData = ti.next();
    QFutureSynchronizer<void> futures;
    while (!templateData.templateIDs.empty()) {
        futures.addFuture(QtConcurrent::run(_janus_create_template, data_path, templateData, gallery));
        templateData = ti.next();
    }
    futures.waitForFinished();
    return JANUS_SUCCESS;
}
