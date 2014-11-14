// stasm_lib_ext.h: experimental extensions to the Stasm library
//                  Please don't use these functions, they may change at any time.

#ifndef STASM_LIB_EXT_H
#define STASM_LIB_EXT_H

extern "C"                   // extended version of stasm_init
int stasm_init_ext(          // call once, at bootup
    const char*  datadir,    // in: directory of face detector files
    int          trace,      // in: 0 normal use, 1 trace to stdout and stasm.log
    void*        detparams); // in: NULL or face detector parameters

extern "C"                   // extended version of stasm_open_image
int stasm_open_image_ext(    // call once per image, detect faces
    const char*  img,        // in: gray image data, top left corner at 0,0
    int          width,      // in: image width
    int          height,     // in: image height
    const char*  imgpath,    // in: image path, used only for err msgs and debug
    int          multiface,  // in: 0=return only one face, 1=allow multiple faces
    int          minwidth,   // in: min face width as percentage of img width
    void*        user);      // in: NULL or pointer to user abort func

extern "C"
int stasm_search_auto_ext(   // extended version of stasm_search_auto
    int*         foundface,  // out: 0=no more faces, 1=found face
    float*       landmarks,  // out: x0, y0, x1, y1, ..., caller must allocate
    float*       estyaw);    // out: NULL or pointer to estimated yaw

#endif // STASM_LIB_EXT_H
