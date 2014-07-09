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

static bool read_buffer(FILE *file, char *buffer, size_t bytes, bool eofAllowed)
{
    size_t bytesRemaining = bytes;
    while (bytesRemaining) {
        const size_t bytesRead = fread(buffer, 1, bytesRemaining, file);
        buffer += bytesRead;
        bytesRemaining -= bytesRead;

        if (feof(file)) {
            if (eofAllowed && (bytesRemaining == bytes))
                return false;
            qFatal("End of file after reading %d of %d bytes.", int(bytes - bytesRemaining), int(bytes));
        }

        if (ferror(file)) {
            perror(NULL);
            qFatal("Error after reading %d of %d bytes.", int(bytes - bytesRemaining), int(bytes));
        }
    }
    return true;
}

void br_iterate_utemplates_file(FILE *file, br_utemplate_callback callback, br_callback_context context, bool parallel)
{
    QFutureSynchronizer<void> futures;
    while (true) {
        br_utemplate t = (br_utemplate) malloc(sizeof(br_universal_template));

        if (!read_buffer(file, (char*) t, sizeof(br_universal_template), true)) {
            free(t);
            return;
        }

        t = (br_utemplate) realloc(t, sizeof(br_universal_template) + t->size);
        read_buffer(file, (char*) &t->data, t->size, false);

        if (parallel) futures.addFuture(QtConcurrent::run(callAndFree, callback, t, context));
        else          callAndFree(callback, t, context);
    }
    futures.waitForFinished();
}

void br_log(const char *message)
{
    qDebug() << qPrintable(QTime::currentTime().toString("hh:mm:ss.zzz")) << "-" << message;
}
