// stasm.h: (nearly) all include files for the Stasm package
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_H
#define STASM_H

static const char* const STASM_VERSION = "4.0.0"
#if MOD_3                     // experimental versions
            "_MOD_3";
#elif MOD_A1
            "_MOD_A1";
#elif MOD_A
            "_MOD_A";
#elif MOD_A_EMU
            "_MOD_A_EMU";
#else
            "";               // released version of Stasm
   #define MOD_1 1
#endif

#define TRACE_IMAGES 0        // 1 to generate debugging images

#if _MSC_VER >= 1200
// disable the following warning:
// opencv2\flann\logger.h(66) : warning C4996: 'fopen': This function may be unsafe
#pragma warning(disable: 4996)
// disable the following warning (x64 builds only)
// opencv2\flann\lsh_table.h(417) : warning C4267: conversion from 'size_t' to ...
#pragma warning(disable:4267)
#endif

#include "opencv/cv.h"

#if _MSC_VER >= 1200
#pragma warning(default:4996) // re-enable the warnings disabled above
#pragma warning(default:4267)
#endif

#if TRACE_IMAGES              // will be 0 unless debugging
#include "opencv/highgui.h"   // need imwrite
#endif

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#if _OPENMP
#include <omp.h>
#endif

#include "misc.h"
#include "print.h"
#include "err.h"
#include "stasm_landmarks.h"
#include "stasm_lib.h"
#include "stasm_lib_ext.h"
#include "landmarks.h"
#include "basedesc.h"
#include "classicdesc.h"
#include "hat.h"
#include "hatdesc.h"
#include "shapehacks.h"
#include "shapemod.h"
#include "asm.h"

#if MOD_1   // released version of Stasm
    #include "../stasm/MOD_1/facedet.h"
    #include "../stasm/MOD_1/initasm.h"
#elif MOD_3 // experimental versions
    #include "../stasm/MOD_3/facedet.h"
    #include "../stasm/MOD_3/initasm.h"
#elif MOD_A1
    #include "../stasm/MOD_A1/facedet.h"
    #include "../stasm/MOD_A1/initasm.h"
#elif MOD_A
    #include "../stasm/MOD_A/facedet.h"
    #include "../stasm/MOD_A/initasm.h"
#elif MOD_A_EMU
    #include "../stasm/MOD_A/facedet.h"
    #include "../stasm/MOD_A/initasm.h"
#else
    error illegal MOD
#endif

#include "eyedet.h"
#include "convshape.h"
#include "eyedist.h"
#include "faceroi.h"
#include "pinstart.h"
#include "shape17.h"
#include "startshape.h"

#endif // STASM_H
