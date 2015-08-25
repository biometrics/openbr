#include <QFile>
#include <QFutureSynchronizer>
#include <QMutex>
#include <QMutexLocker>
#include <QtConcurrent>
#include <cstdlib>
#include <cstring>

#include "universal_template.h"

br_utemplate br_new_utemplate(int32_t algorithmID, uint32_t frame, int32_t x, int32_t y, uint32_t width, uint32_t height, float confidence, uint32_t personID, const char *metadata, const char *featureVector, uint32_t fvSize)
{
    const uint32_t mdSize = strlen(metadata) + 1;
    br_utemplate utemplate = (br_utemplate) malloc(sizeof(br_universal_template) + mdSize + fvSize);
    utemplate->algorithmID = algorithmID;
    utemplate->frame = frame;
    utemplate->x = x;
    utemplate->y = y;
    utemplate->width = width;
    utemplate->height = height;
    utemplate->confidence = confidence;
    utemplate->personID = personID;
    utemplate->mdSize = mdSize;
    utemplate->fvSize = fvSize;
    memcpy(reinterpret_cast<char*>(utemplate+1) + 0,      metadata     , mdSize);
    memcpy(reinterpret_cast<char*>(utemplate+1) + mdSize, featureVector, fvSize);
    return utemplate;
}

void br_free_utemplate(br_const_utemplate utemplate)
{
    free((void*) utemplate);
}

void br_append_utemplate(FILE *file, br_const_utemplate utemplate)
{
    fwrite(utemplate, sizeof(br_universal_template) + utemplate->mdSize + utemplate->fvSize, 1, file);
}

void br_iterate_utemplates(br_const_utemplate begin, br_const_utemplate end, br_utemplate_callback callback, br_callback_context context)
{
    while (begin != end) {
        callback(begin, context);
        begin = reinterpret_cast<br_const_utemplate>(reinterpret_cast<const char*>(begin) + sizeof(br_universal_template) + begin->mdSize + begin->fvSize);
        if (begin > end)
            qFatal("Overshot end of buffer");
    }
}

static void callAndFree(br_utemplate_callback callback, br_utemplate t, br_callback_context context)
{
    callback(t, context);
    free(t);
}

static bool read_buffer(FILE *file, char *buffer, size_t bytes)
{
    size_t bytesRemaining = bytes;
    while (bytesRemaining) {
        const size_t bytesRead = fread(buffer, 1, bytesRemaining, file);
        buffer += bytesRead;
        bytesRemaining -= bytesRead;

        if (feof(file)) {
            if ((bytesRemaining != bytes) && !fseek(file, bytesRemaining - bytes, SEEK_CUR)) // Try to rewind
                bytesRemaining = bytes;
            if (bytesRemaining == bytes)
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

int br_iterate_utemplates_file(FILE *file, br_utemplate_callback callback, br_callback_context context, bool parallel)
{
    int count = 0;
    QFutureSynchronizer<void> futures;
    while (true) {
        br_utemplate t = (br_utemplate) malloc(sizeof(br_universal_template));

        if (!read_buffer(file, (char*) t, sizeof(br_universal_template))) {
            free(t);
            break;
        }

        t = (br_utemplate) realloc(t, sizeof(br_universal_template) + t->mdSize + t->fvSize);
        if (!read_buffer(file, (char*) &t->data, t->mdSize + t->fvSize)) {
            free(t);

            // Try to rewind header read
            if (fseek(file, -long(sizeof(br_universal_template)), SEEK_CUR))
                qFatal("Unable to recover from partial template read!");

            break;
        }

        if (parallel) futures.addFuture(QtConcurrent::run(callAndFree, callback, t, context));
        else          callAndFree(callback, t, context);
        count++;
    }
    return count;
}

void br_log(const char *message)
{
    qDebug() << qPrintable(QTime::currentTime().toString("hh:mm:ss.zzz")) << "-" << message;
}
