#include <QFile>
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

void br_append_utemplate_contents(FILE *file, const int8_t *imageID, const int8_t *templateID, int32_t algorithmID, uint32_t size, const int8_t *data)
{
    QFile qFile;
    qFile.open(file, QFile::WriteOnly | QFile::Append);
    qFile.write((const char*) imageID, 16);
    qFile.write((const char*) templateID, 16);
    qFile.write((const char*) &algorithmID, 4);
    qFile.write((const char*) &size, 4);
    qFile.write((const char*) data, size);
}
