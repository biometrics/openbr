#include <QFile>
#include <QFutureSynchronizer>
#include <QMutex>
#include <QMutexLocker>
#include <QtConcurrent>
#include <cstdlib>
#include <cstring>

#include "universal_template.h"

br_utemplate br_new_utemplate(const int8_t *imageID, const int8_t *templateID, int32_t algorithmID, uint32_t size, const int8_t *data)
{
    br_utemplate utemplate = (br_utemplate) malloc(sizeof(br_universal_template) + size);
    memcpy(utemplate->imageID, imageID, 16);
    memcpy(utemplate->templateID, templateID, 16);
    utemplate->algorithmID = algorithmID;
    utemplate->size = size;
    memcpy(utemplate+1, data, size);
    return utemplate;
}

void br_free_utemplate(br_const_utemplate utemplate)
{
    free((void*) utemplate);
}

void br_append_utemplate(FILE *file, br_const_utemplate utemplate)
{
    br_append_utemplate_contents(file, utemplate->imageID, utemplate->templateID, utemplate->algorithmID, utemplate->size, utemplate->data);
}

void br_append_utemplate_contents(FILE *file, const unsigned char *imageID, const unsigned char *templateID, int32_t algorithmID, uint32_t size, const unsigned char *data)
{
    static QMutex lock;
    QMutexLocker locker(&lock);

    fwrite(imageID, 16, 1, file);
    fwrite(templateID, 16, 1, file);
    fwrite(&algorithmID, 4, 1, file);
    fwrite(&size, 4, 1, file);
    fwrite(data, 1, size, file);
}

void br_iterate_utemplates(br_const_utemplate begin, br_const_utemplate end, br_utemplate_callback callback, br_callback_context context)
{
    while (begin != end) {
        callback(begin, context);
        begin = reinterpret_cast<br_const_utemplate>(reinterpret_cast<const char*>(begin) + sizeof(br_universal_template) + begin->size);
    }
}

static void callAndFree(br_utemplate_callback callback, br_utemplate t, br_callback_context context)
{
    callback(t, context);
    free(t);
}

void br_iterate_utemplates_file(FILE *file, br_utemplate_callback callback, br_callback_context context, bool parallel)
{
    QFutureSynchronizer<void> futures;
    while (true) {
        br_utemplate t = (br_utemplate) malloc(sizeof(br_universal_template));

        int bytesRemaining = sizeof(br_universal_template);
        while (bytesRemaining > 0) {
            bytesRemaining -= fread(reinterpret_cast<char*>(t) + sizeof(br_universal_template) - bytesRemaining, 1, bytesRemaining, file);

            if (feof(file)) {
                if (bytesRemaining == sizeof(br_universal_template)) {
                    free(t);
                    return;
                } else {
                    qFatal("Unexpected end of file when reading template metadata.");
                }
            }

            if (ferror(file)) {
                perror(NULL);
                qFatal("Error while reading template metadata.");
            }
        }

        t = (br_utemplate) realloc(t, sizeof(br_universal_template) + t->size);
        bytesRemaining = t->size;
        while (bytesRemaining > 0) {
            bytesRemaining -= fread(&t->data[t->size - bytesRemaining], 1, bytesRemaining, file);

            if (feof(file))
                qFatal("Unexpected end of file when reading template data.");

            if (ferror(file)) {
                perror(NULL);
                qFatal("Error while reading template data.");
            }
        }

        if (parallel) futures.addFuture(QtConcurrent::run(callAndFree, callback, t, context));
        else          callAndFree(callback, t, context);
    }
    futures.waitForFinished();
}
