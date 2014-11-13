#include <QFile>
#include <QFutureSynchronizer>
#include <QMutex>
#include <QMutexLocker>
#include <QtConcurrent>
#include <cstdlib>
#include <cstring>

#include "universal_template.h"

br_utemplate br_new_utemplate(const int8_t *imageID, int32_t algorithmID, size_t x, size_t y, size_t width, size_t height, double label, const char *url, const char *fv, uint32_t fvSize)
{
    const uint32_t urlSize = strlen(url) + 1;
    br_utemplate utemplate = (br_utemplate) malloc(sizeof(br_universal_template) + urlSize + fvSize);
    memcpy(utemplate->imageID, imageID, 16);
    utemplate->algorithmID = algorithmID;
    utemplate->x = x;
    utemplate->y = y;
    utemplate->width = width;
    utemplate->height = height;
    utemplate->label = label;
    utemplate->urlSize = urlSize;
    utemplate->fvSize = fvSize;
    memcpy(reinterpret_cast<char*>(utemplate+1) + 0,       url , urlSize);
    memcpy(reinterpret_cast<char*>(utemplate+1) + urlSize, fv, fvSize);
    return utemplate;
}

void br_free_utemplate(br_const_utemplate utemplate)
{
    free((void*) utemplate);
}

void br_append_utemplate(FILE *file, br_const_utemplate utemplate)
{
    fwrite(utemplate, sizeof(br_universal_template) + utemplate->urlSize + utemplate->fvSize, 1, file);
}

void br_iterate_utemplates(br_const_utemplate begin, br_const_utemplate end, br_utemplate_callback callback, br_callback_context context)
{
    while (begin != end) {
        callback(begin, context);
        begin = reinterpret_cast<br_const_utemplate>(reinterpret_cast<const char*>(begin) + sizeof(br_universal_template) + begin->urlSize + begin->fvSize);
        if (begin > end)
            qFatal("Overshot end of buffer");
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

        t = (br_utemplate) realloc(t, sizeof(br_universal_template) + t->urlSize + t->fvSize);
        read_buffer(file, (char*) &t->data, t->urlSize + t->fvSize, false);

        if (parallel) futures.addFuture(QtConcurrent::run(callAndFree, callback, t, context));
        else          callAndFree(callback, t, context);
    }
    futures.waitForFinished();
}

void br_log(const char *message)
{
    qDebug() << qPrintable(QTime::currentTime().toString("hh:mm:ss.zzz")) << "-" << message;
}
