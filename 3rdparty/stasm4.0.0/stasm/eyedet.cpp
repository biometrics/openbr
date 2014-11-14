// eyedet.cpp: interface to OpenCV eye and mouth detectors
//
// Copyright (C) 2005-2013, Stephen Milborrow

#include "stasm.h"

namespace stasm
{
//static cv::CascadeClassifier leye_det_g;  // left eye detector
//static cv::CascadeClassifier reye_det_g;  // right eye detector
//static cv::CascadeClassifier mouth_det_g; // mouth detector

//-----------------------------------------------------------------------------

// Return the region of the face we search for the left or right eye.
// Return rect of width=0 if eye must not be searched for (outer eyes in side views).
// We reduce false positives and save time by searching in only part of the face.
// The entire eye box must fall in this region, not just the center of the eye.
// The magic numbers below were found empirically to give good
// results in informal tests.  They reduce the number of false positives
// in the forehead, eyebrows, nostrils, and mouth.

static Rect EyeSearchRect(
    EYAW        eyaw,         // in
    const Rect& facerect,     // in
    const bool  is_right_eye) // in: true for right eye, false for left eye
{
    Rect rect = facerect;
    int width = facerect.width;
    switch (eyaw)
    {
        case EYAW00:                        // frontal model
            if (is_right_eye)
                rect.x += width / 3; // don't search left third of face
            rect.width -= width / 3; // or right third
            rect.height = cvRound(.6 * facerect.height); // don't search lower part of face
            break;
        case EYAW_22:                       // left facing three-quarter model
            if (is_right_eye)               // inner eye
            {
                rect.x += cvRound(.4 * width);
                rect.width = cvRound(.5 * width);
            }
            else                            // outer eye
            {
                rect.x += cvRound(.1 * width);
                rect.width = cvRound(.5 * width);
            }
            rect.height = cvRound(.5 * facerect.height);
            break;
        case EYAW22:                        // right facing three-quarter model
            if (is_right_eye)               // outer eye
            {
                rect.x += cvRound(.4 * width);
                rect.width = cvRound(.5 * width);
            }
            else                            // inner eye
            {
                rect.x += cvRound(.1 * width);
                rect.width = cvRound(.5 * width);
            }
            rect.height = cvRound(.5 * facerect.height);
            break;
        case EYAW_45:                       // left facing three-quarter model
            if (is_right_eye)               // inner eye
            {
                rect.x += cvRound(.4 * width);
                rect.width = cvRound(.5 * width);
                rect.height = cvRound(.5 * facerect.height);
            }
            else                            // outer eye
                rect.width = rect.height = 0;
            break;
        case EYAW45:                        // right facing three-quarter model
            if (is_right_eye)               // outer eye
                rect.width = rect.height = 0;
            else                            // inner eye
            {
                rect.x += cvRound(.1 * width);
                rect.width = cvRound(.5 * width);
                rect.height = cvRound(.5 * facerect.height);
            }
            break;
        default:
            Err("EyeSearchRect: Invalid eyaw %d", eyaw);
            break;
    }
    rect.width  = MAX(0, rect.width);
    rect.height = MAX(0, rect.height);
    return rect;
}

// Get adjustment for position of mouth, based on model type and eye angle.

static void MouthRectShift(
    int&            ixshift,         // out
    int&            iyshift,         // out
    EYAW            eyaw,            // in
    int             facerect_width,  // in
    int             facerect_height, // in
    int             ileft_best,      // in
    int             iright_best,     // in
    const vec_Rect& leyes,           // in
    const vec_Rect& reyes)           // in
{
    double xshift = 0, yshift = 0;
    switch (eyaw)
    {
        case EYAW00: // frontal model
            break;
        case EYAW_45: // left facing three-quarter model
            xshift -= .04 * facerect_width;
            break;
        case EYAW_22: // left facing three-quarter model
            xshift -= .03 * facerect_width;
            break;
        case EYAW22: // right facing three-quarter model
            xshift += .03 * facerect_width;
            break;
        case EYAW45: // right facing three-quarter model
            xshift += .04 * facerect_width;
            break;
        default:
            Err("GeMouthRect: Invalid eyaw %d", eyaw);
            break;
    }

    if (ileft_best != -1 && iright_best != -1)   // got both eyes?
    {
        // get center of eye boxes to get eye angle
        const int xleft  = leyes[ileft_best].x  + leyes[ileft_best].width/2;
        const int yleft  = leyes[ileft_best].y  + leyes[ileft_best].height/2;
        const int xright = reyes[iright_best].x + reyes[iright_best].width/2;
        const int yright = reyes[iright_best].y + reyes[iright_best].height/2;
        double theta = -atan2(double(yright - yleft), double(xright - xleft));
        // move the mouth in the direction of rotation
        xshift += .3 * facerect_height * tan(theta);
        // as the face rotates, the mouth moves up the page
        yshift -= .1 * facerect_height * ABS(tan(theta));
    }
    ixshift = cvRound(xshift);
    iyshift = cvRound(yshift);
}

static Rect MouthRect(           // will search for mouth in this rectangle
    const Rect&     facerect,    // in
    EYAW            eyaw,        // in
    int             ileft_best,  // in: index of best left eye, -1 if none
    int             iright_best, // in: index of best right eye, -1 if none
    const vec_Rect& leyes,       // in: left eyes found by eye detector
    const vec_Rect& reyes)       // in: right eyes found by eye detector
{
    Rect rect = facerect;

    int ixshift, iyshift;
    MouthRectShift(ixshift, iyshift,
                   eyaw, facerect.width, facerect.height,
                   ileft_best, iright_best, leyes, reyes);

    rect.x += cvRound(.2  * facerect.width) + ixshift;

    rect.width = MAX(1, cvRound(.6  * facerect.width));

    switch (eyaw)
    {
        case EYAW00: // frontal model
            rect.y += cvRound(.64 * facerect.height);
            break;
        case EYAW_45: // left facing three-quarter model
            rect.y += cvRound(.55 * facerect.height);
            break;
        case EYAW_22: // left facing three-quarter model
            rect.y += cvRound(.55 * facerect.height);
            break;
        case EYAW22: // right facing three-quarter model
            rect.y += cvRound(.55 * facerect.height);
            break;
        case EYAW45: // right facing three-quarter model
            rect.y += cvRound(.55 * facerect.height);
            break;
        default:
            Err("MouthRect: Invalid eyaw %d", eyaw);
            break;
    }
    rect.y += iyshift;
    rect.height = cvRound(.42 * facerect.height);
    rect.width  = MAX(0, rect.width);
    rect.height = MAX(0, rect.height);
    return rect;
}

bool NeedMouth(
    const vec_Mod& mods) // in: the ASM model(s)
{
    for (int imod = 0; imod < NSIZE(mods); imod++)
        if (mods[imod]->Estart_() == ESTART_EYE_AND_MOUTH)
            return true;
    return false;
}

// Possibly open OpenCV eye detectors and mouth detector.  We say "possibly" because
// the eye and mouth detectors will actually only be opened if any model in mods
// actually needs them.  That is determined by the model's estart field.

void OpenEyeMouthDetectors(
    const vec_Mod& mods,    // in: the ASM models (used to see if we need eyes or mouth)
    const char*    datadir) // in
{
    (void) mods;
    (void) datadir;
    /*
    static bool needeyes = true; // static for efficiency
    // we need the eyes if the estart field of any model
    // is ESTART_EYES or ESTART_EYE_AND_MOUTH
    needeyes = false;
    for (int imod = 0; imod < NSIZE(mods); imod++)
        if (mods[imod]->Estart_() == ESTART_EYES ||
                mods[imod]->Estart_() == ESTART_EYE_AND_MOUTH)
            needeyes = true;
    if (needeyes)
    {
        // I tried all the eye XML files that come with OpenCV 2.1 and found that
        // the files used below give the best results.  The other eye XML files
        // often failed to detect eyes, even with EYE_MIN_NEIGHBORS=1.
        //
        // In the XML filenames, "left" was verified empirically by me to respond
        // to the image left (not the subject's left).  I tested this on the on
        // the MUCT and BioID sets: haarcascade_mcs_lefteye.xml finds more eyes
        // on the viewer's left than it finds on the right (milbo Lusaka Dec 2011).

        //OpenDetector(leye_det_g,  "haarcascade_mcs_lefteye.xml",  datadir);
        //OpenDetector(reye_det_g,  "haarcascade_mcs_righteye.xml", datadir);
    }
    static bool needmouth = true; // static for efficiency
    // we need the eyes if the estart field of any model is ESTART_EYE_AND_MOUTH
    needmouth = false;
    for (int imod = 0; imod < NSIZE(mods); imod++)
        if (mods[imod]->Estart_() == ESTART_EYE_AND_MOUTH)
            needmouth = true;
    if (needmouth)
        //OpenDetector(mouth_det_g, "haarcascade_mcs_mouth.xml", datadir);*/
}

static void DetectAllEyes(
    vec_Rect&    leyes,    // out
    vec_Rect&    reyes,    // out
    const Image& img,      // in
    EYAW         eyaw,     // in
    const Rect&  facerect, // in
    StasmCascadeClassifier cascade)
{
    // 1.2 is 40ms faster than 1.1 but finds slightly fewer eyes
    static const double EYE_SCALE_FACTOR   = 1.2;
    static const int    EYE_MIN_NEIGHBORS  = 3;
    static const int    EYE_DETECTOR_FLAGS = 0;

    Rect leftrect(EyeSearchRect(eyaw, facerect, false));

    if (leftrect.width)
        leyes = Detect(img, &cascade.leftEyeCascade, &leftrect,
                       EYE_SCALE_FACTOR, EYE_MIN_NEIGHBORS, EYE_DETECTOR_FLAGS,
                       facerect.width / 10);

    Rect rightrect(EyeSearchRect(eyaw, facerect, true));

    if (rightrect.width)
        reyes = Detect(img, &cascade.rightEyeCascade, &rightrect,
                       EYE_SCALE_FACTOR, EYE_MIN_NEIGHBORS, EYE_DETECTOR_FLAGS,
                       facerect.width / 10);
}

static void DetectAllMouths(
    vec_Rect&       mouths,      // out
    const Image&    img,         // in
    EYAW            eyaw,        // in
    const Rect&     facerect,    // in
    int             ileft_best,  // in
    int             iright_best, // in
    const vec_Rect& leyes,       // in
    const vec_Rect& reyes,       // in
    cv::CascadeClassifier cascade)
{
    static const double MOUTH_SCALE_FACTOR   = 1.2; // less false pos with 1.2 than 1.1
    static const int    MOUTH_MIN_NEIGHBORS  = 5;   // less false pos with 5 than 3
    static const int    MOUTH_DETECTOR_FLAGS = 0;

    Rect mouth_rect(MouthRect(facerect,
                              eyaw, ileft_best, iright_best, leyes, reyes));

    mouths =
        Detect(img, &cascade, &mouth_rect,
               MOUTH_SCALE_FACTOR, MOUTH_MIN_NEIGHBORS, MOUTH_DETECTOR_FLAGS,
               facerect.width / 10);
}

// Return the region of the face which the _center_ of an eye must be for
// the eye to be considered valid.  This is a subset of the region we
// search for eyes (as returned by EyeSearchRect, which must be big
// enough to enclose the _entire_ eye box).

static Rect EyeInnerRect(
    EYAW        eyaw,        // in
    const Rect& facerect)    // in
{
    Rect rect = facerect;
    switch (eyaw)
    {
        case EYAW00: // frontal model
            rect.x     += cvRound(.1 * facerect.width);
            rect.width  = cvRound(.8 * facerect.width);
            rect.y     += cvRound(.2 * facerect.height);
            rect.height = cvRound(.28 * facerect.height);
            break;
        case EYAW_45: // left facing three-quarter model
            rect.x     += cvRound(.4 * facerect.width);
            rect.width  = cvRound(.5 * facerect.width);
            rect.y     += cvRound(.20 * facerect.height);
            rect.height = cvRound(.25 * facerect.height);
            break;
        case EYAW_22: // left facing three-quarter model
            rect.x     += cvRound(.1 * facerect.width);
            rect.width  = cvRound(.8 * facerect.width);
            rect.y     += cvRound(.20 * facerect.height);
            rect.height = cvRound(.25 * facerect.height);
            break;
        case EYAW22: // right facing three-quarter model
            rect.x     += cvRound(.1 * facerect.width);
            rect.width  = cvRound(.8 * facerect.width);
            rect.y     += cvRound(.20 * facerect.height);
            rect.height = cvRound(.25 * facerect.height);
            break;
        case EYAW45: // right facing three-quarter model
            rect.x     += cvRound(.1 * facerect.width);
            rect.width  = cvRound(.5 * facerect.width);
            rect.y     += cvRound(.20 * facerect.height);
            rect.height = cvRound(.25 * facerect.height);
            break;
        default:
            Err("EyeInnerRect: Invalid eyaw %d", eyaw);
            break;
    }
    rect.width  = MAX(0, rect.width);
    rect.height = MAX(0, rect.height);
    return rect;
}

// Is the horizontal overlap between the LeftEye and RightEye rectangles no
// more than 10% and is the horizontal distance between the edges of the
// eyes no more than the eye width.

static bool IsEyeHorizOk(
    const Rect& left,         // in
    const Rect& right)        // in
{
    return left.x + left.width - right.x   <= .1 * left.width &&
           right.x - (left.x + left.width) <= left.width;
}

static bool VerticalOverlap( // do the two eyes overlap vertically?
    const Rect& left,        // in
    const Rect& right)       // in
{
    const int topleft = left.y + left.height;
    const int topright = right.y + right.height;

    return (left.y   >= right.y && left.y   <= right.y + right.height) ||
           (topleft  >= right.y && topleft  <= right.y + right.height) ||
           (right.y  >= left.y  && right.y  <= left.y  + left.height)  ||
           (topright >= left.y  && topright <= left.y  + left.height);
}


// Is the center of rect within the enclosing rect?

static bool InRect(
    const Rect& rect,      // in
    const Rect& enclosing) // in
{
    int x = rect.x + rect.width / 2;  // center of rectangle
    int y = rect.y + rect.height / 2;

    return x >= enclosing.x &&
           x <= enclosing.x + enclosing.width &&
           y >= enclosing.y &&
           y <= enclosing.y + enclosing.height;
}

// Return the indices of the best left and right eye in the list of eyes.
// returned by the feature detectors.
// The heuristic in in detail (based on looking at images produced):
// Find the left and right eye that
//  (i)   are both in eye_inner_rect
//  (ii)  don't overlap horizontally by more than 10%
//  (ii)  overlap vertically.
//  (iii) have the largest total width.
//  (iv)  if frontal have an intereye dist at least .25 * eye_inner_rect width

static void SelectEyes(
    int&            ileft_best,     // out: index into leyes, -1 if none
    int&            iright_best,    // out: index into reyes, -1 if none
    EYAW            eyaw,           // in
    const vec_Rect& leyes,          // in: left eyes found by detectMultiScale
    const vec_Rect& reyes,          // in: right eyes found by detectMultiScale
    const Rect&     eye_inner_rect) // in: center of the eye must be in this region
{
    ileft_best = iright_best = -1; // assume will return no eyes
    int min_intereye = eyaw == EYAW00? cvRound(.25 * eye_inner_rect.width): 0;
    int maxwidth = 0; // combined width of both eye boxes
    int ileft, iright;
    Rect left, right;

    // this part of the code will either select both eyes or no eyes

    for (ileft = 0; ileft < NSIZE(leyes); ileft++)
    {
        left = leyes[ileft];
        if (InRect(left, eye_inner_rect))
        {
            for (iright = 0; iright < NSIZE(reyes); iright++)
            {
                right = reyes[iright];
                if (InRect(right, eye_inner_rect) &&
                    IsEyeHorizOk(left, right) &&
                    right.x - left.x >= min_intereye &&
                    VerticalOverlap(left, right))
                {
                    int total_width = left.width + right.width;
                    if (total_width > maxwidth)
                    {
                        maxwidth = total_width;
                        ileft_best = ileft;
                        iright_best = iright;
                    }
                }
            }
        }
    }
    if (ileft_best == -1 && iright_best == -1)
    {
        // The above loops failed to find a left and right eye in correct
        // relationship to each other.  So simply select largest left eye and
        // largest right eye (but make sure that they are in the eye_inner_rect).

        int max_left_width = 0;
        for (ileft = 0; ileft < NSIZE(leyes); ileft++)
        {
            left = leyes[ileft];
            if (InRect(left, eye_inner_rect))
            {
                if (left.width > max_left_width)
                {
                    max_left_width = left.width;
                    ileft_best = ileft;
                }
            }
        }
        int max_right_width = 0;
        for (iright = 0; iright < NSIZE(reyes); iright++)
        {
            right = reyes[iright];
            if (InRect(right, eye_inner_rect))
            {
                if (right.width > max_right_width)
                {
                    max_right_width = right.width;
                    iright_best = iright;
                }
            }
        }
        // One final check (for vr08m03.bmp) -- if the two largest eyes overlap
        // too much horizontally then discard the smaller eye.

        if (ileft_best != -1 && iright_best != -1)
        {
            left = leyes[ileft_best];
            right = reyes[iright_best];
            if (!IsEyeHorizOk(left, right) || right.x - left.x < min_intereye)
            {
                if (max_right_width > max_left_width)
                    ileft_best = -1;
                else
                    iright_best = -1;
            }
        }
    }
}

// The values below are fairly conservative: for the ASM start shape,
// it's better to not find a mouth than to find an incorrect mouth.

static Rect MouthInnerRect(
    const Rect&     facerect,    // in
    EYAW            eyaw,        // in
    int             ileft_best,  // in: index of best left eye, -1 if none
    int             iright_best, // in: index of best right eye, -1 if none
    const vec_Rect& leyes,       // in: left eyes found by eye detector
    const vec_Rect& reyes)       // in: right eyes found by eye detector
{
    Rect rect = facerect;
    double width = (eyaw == EYAW00? .12: .20) * facerect.width;
    double height = .30 * facerect.height;

    int ixshift, iyshift;
    MouthRectShift(ixshift, iyshift,
                   eyaw, facerect.width, facerect.height,
                   ileft_best, iright_best, leyes, reyes);

    rect.x += cvRound(.50 * (facerect.width - width)) + ixshift;

    rect.width  =  cvRound(width);

    switch (eyaw)
    {
        case EYAW00: // frontal model
            rect.y += cvRound(.7 * facerect.height);
            break;
        case EYAW_45: // left facing three-quarter model
            rect.y += cvRound(.65 * facerect.height);
            break;
        case EYAW_22: // left facing three-quarter model
            rect.y += cvRound(.65 * facerect.height);
            break;
        case EYAW22: // right facing three-quarter model
            rect.y += cvRound(.65 * facerect.height);
            break;
        case EYAW45: // right facing three-quarter model
            rect.y += cvRound(.65 * facerect.height);
            break;
        default:
            Err("MouthInnerRect: Invalid eyaw %d", eyaw);
            break;
    }
    rect.y += iyshift;
    rect.height = cvRound(height);
    rect.width  = MAX(0, rect.width);
    rect.height = MAX(0, rect.height);
    return rect;
}

// The OpenCV mouth detector biases the position of the mouth downward (wrt the
// center of the mouth determined by manual landmarking).  Correct that here.

static int MouthVerticalShift(
    const int       ileft_best,   // in
    const int       iright_best,  // in
    const int       imouth_best,  // in
    const vec_Rect& leyes,        // in
    const vec_Rect& reyes,        // in
    const vec_Rect& mouths)       // in
{
    double shift = 0;
    if (ileft_best != -1 && iright_best != -1) // got both eyes?
    {
        CV_Assert(imouth_best != -1);
        // get eye mouth distance: first get center of both eyes
        const double xleft  = leyes[ileft_best].x  + leyes[ileft_best].width   / 2;
        const double yleft  = leyes[ileft_best].y  + leyes[ileft_best].height  / 2;
        const double xright = reyes[iright_best].x + reyes[iright_best].width  / 2;
        const double yright = reyes[iright_best].y + reyes[iright_best].height / 2;
        const double eyemouth =
            PointDist((xleft + xright) / 2,(yleft + yright) / 2,
                      mouths[imouth_best].x, mouths[imouth_best].y);
        static const double MOUTH_VERT_ADJUST = -0.050; // neg to shift up
        shift = MOUTH_VERT_ADJUST * eyemouth;
    }
    return cvRound(shift);
}

// Return the indices of the best mouth in the list of mouths

static void SelectMouth(
    int&            imouth_best,      // out: index into mouths, -1 if none
    int             ileft_best,       // in: index of best left eye, -1 if none
    int             iright_best,      // in: index of best right eye, -1 if none
    const vec_Rect& leyes,            // in: left eyes found by eye detector
    const vec_Rect& reyes,            // in: right eyes found by eye detector
    const vec_Rect& mouths,           // in: left eyes found by eye detector
    const Rect&     mouth_inner_rect) // in: center of mouth must be in this region
{
    CV_Assert(!mouths.empty());
    imouth_best = -1;

    // if only one mouth, use it
    if (NSIZE(mouths) == 1 && InRect(mouths[0], mouth_inner_rect))
        imouth_best = 0;
    else
    {
        // More than one mouth: selected the lowest mouth to avoid
        // "nostril mouths".  But to avoid "chin mouths", the mouth
        // must also meet the following criteria:
        //   i)  it must be wider than the .7 * smallest eye width
        //   ii) it must be not much narrower than widest mouth.

        int minwidth = 0;
        if (ileft_best != -1)
            minwidth = leyes[ileft_best].width;
        if (iright_best != -1)
            minwidth = MIN(minwidth, reyes[iright_best].width);
        minwidth = cvRound(.7 * minwidth);

        // find widest mouth
        int maxwidth = minwidth;
        for (int imouth = 0; imouth < NSIZE(mouths); imouth++)
        {
            const Rect mouth = mouths[imouth];
            if (InRect(mouth, mouth_inner_rect) && mouth.width > maxwidth)
            {
                maxwidth = mouth.width;
                imouth_best = imouth;
            }
        }
        // choose lowest mouth that is at least .84 the width of widest
        minwidth = MAX(minwidth, cvRound(.84 * maxwidth));
        int ymin = int(-1e5);
        for (int imouth = 0; imouth < NSIZE(mouths); imouth++)
        {
            const Rect mouth = mouths[imouth];
            if (InRect(mouth, mouth_inner_rect) &&
                mouth.y + mouth.height / 2 > ymin &&
                mouth.width > minwidth)
            {
                ymin = mouth.y + mouth.height / 2;
                imouth_best = imouth;
            }
        }
    }
}

static void TweakMouthPosition(
    vec_Rect&       mouths,      // io
    const vec_Rect& leyes,       // in
    const vec_Rect& reyes,       // in
    const int       ileft_best,  // in
    const int       iright_best, // in
    const int       imouth_best, // in
    const DetPar&   detpar)      // in

{
    mouths[imouth_best].y += // move mouth up to counteract OpenCV mouth bias
         MouthVerticalShift(ileft_best, iright_best, imouth_best,
                            leyes, reyes, mouths);

    // If face pose is strong three-quarter, move mouth
    // out to counteract OpenCV mouth detector bias.

    if (detpar.eyaw == EYAW_45)
        mouths[imouth_best].x -= cvRound(.06 * detpar.width);
    else if (detpar.eyaw == EYAW45)
        mouths[imouth_best].x += cvRound(.06 * detpar.width);
}

static void RectToImgFrame(
    double&     x,          // out: center of feature
    double&     y,          // out: center of feature
    const Rect& featrect)   // in
{
    x = featrect.x + featrect.width / 2;
    y = featrect.y + featrect.height / 2;
}

void DetectEyesAndMouth(  // use OpenCV detectors to find the eyes and mouth
    DetPar&       detpar, // io: eye and mouth fields updated, other fields untouched
    const Image&  img,    // in: ROI around face (already rotated if necessary)
    StasmCascadeClassifier cascade)
{
    Rect facerect(cvRound(detpar.x - detpar.width/2),
                  cvRound(detpar.y - detpar.height/2),
                  cvRound(detpar.width),
                  cvRound(detpar.height));

    // possibly get the eyes

    detpar.lex = detpar.ley = INVALID; // mark eyes as unavailable
    detpar.rex = detpar.rey = INVALID;
    vec_Rect leyes, reyes;
    int ileft_best = -1, iright_best = -1; // indices into leyes and reyes

    DetectAllEyes(leyes, reyes, img, detpar.eyaw, facerect, cascade);

    SelectEyes(ileft_best, iright_best, detpar.eyaw, leyes, reyes, EyeInnerRect(detpar.eyaw, facerect));

    if (ileft_best >= 0)
        RectToImgFrame(detpar.lex, detpar.ley,
                       leyes[ileft_best]);

    if (iright_best >= 0)
        RectToImgFrame(detpar.rex, detpar.rey,
                       reyes[iright_best]);

    // possibly get the mouth

    detpar.mouthx = detpar.mouthy = INVALID;  // mark mouth as unavailable
    vec_Rect mouths;
    DetectAllMouths(mouths,
                    img, detpar.eyaw, facerect,
                    ileft_best, iright_best, leyes, reyes, cascade.mouthCascade);

    if (!mouths.empty())
    {
        int imouth_best = -1;

        SelectMouth(imouth_best,
                    ileft_best, iright_best, leyes, reyes, mouths,
                    MouthInnerRect(facerect, detpar.eyaw,
                                   ileft_best, iright_best, leyes, reyes));

        if (imouth_best >= 0)
        {
            TweakMouthPosition(mouths,
                               leyes, reyes, ileft_best, iright_best,
                               imouth_best, detpar);

            RectToImgFrame(detpar.mouthx, detpar.mouthy,
                           mouths[imouth_best]);
        }
    }
}

} // namespace stasm
