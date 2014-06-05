#include <QFile>
#include <QMutex>
#include <QMutexLocker>
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

void br_append_utemplate_contents(FILE *file, const int8_t *imageID, const int8_t *templateID, int32_t algorithmID, uint32_t size, const unsigned char *data)
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
        begin = reinterpret_cast<br_const_utemplate>(reinterpret_cast<const char*>(begin) + sizeof(br_const_utemplate) + begin->size);
    }
}

void br_iterate_utemplates_file(FILE *file, br_utemplate_callback callback, br_callback_context context)
{
    while (!feof(file)) {
        br_utemplate t = (br_utemplate) malloc(sizeof(br_universal_template));

        if (fread(t, sizeof(br_universal_template), 1, file) > 0) {
            t = (br_utemplate) realloc(t, sizeof(br_universal_template) + t->size);
            if (fread(t+1, 1, t->size, file) != t->size)
                qFatal("Unexepected EOF when reading universal template data.");
            callback(t, context);
        }

        free(t);
    }
}
