/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2014 Noblis                                                     *
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License");           *
 * you may not use this file except in compliance with the License.          *
 * You may obtain a copy of the License at                                   *
 *                                                                           *
 *     http://www.apache.org/licenses/LICENSE-2.0                            *
 *                                                                           *
 * Unless required by applicable law or agreed to in writing, software       *
 * distributed under the License is distributed on an "AS IS" BASIS,         *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 * See the License for the specific language governing permissions and       *
 * limitations under the License.                                            *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <cstdlib>
#include <cstring>
#include <openbr/universal_template.h>

static void help()
{
    printf("br-print [args]\n"
           "================\n"
           "* __stdin__  - Templates\n"
           "* __stdout__ - Template _data_\n"
           "\n"
           "_br-print_ extracts template data.\n");
}

static void print_utemplate(br_const_utemplate utemplate, br_callback_context)
{
    fwrite(utemplate->data, 1, utemplate->size, stdout);
    printf("\n");
}

int main(int argc, char *argv[])
{
    for (int i=1; i<argc; i++)
        if      (!strcmp(argv[i], "-help")) { help(); exit(EXIT_SUCCESS); }

    br_iterate_utemplates_file(stdin, print_utemplate, NULL);
    return EXIT_SUCCESS;
}
