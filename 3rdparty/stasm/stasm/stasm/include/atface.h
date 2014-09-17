// atface.h: face point attributes
//
// These attributes are used when training new models (they appear in
// LANDMARK_INFO_TAB).  They are also sometimes used when testing Stasm.
// They are not needed for doing landmark searches.
//
// Some of these bits apply to the entire image (AT_BadImg); other apply
// only to a specific points (AT_Glasses).
//
// In shape files, the bits appear in the in the "tag" preceding each shape.
// For example
// 0004 d0145
// means that the face in image d0145.jpg is wearing glasses.

#ifndef STASM_ATFACE_HPP
#define STASM_ATFACE_HPP

static const char* const AT_NAMES[] = // used only for information during training
{
    "Male",       // AT_Male       0x0001
    "Child",      // AT_Child      0x0002
    "Glasses",    // AT_Glasses    0x0004
    "Beard",      // AT_Beard      0x0008
    "Mustache",   // AT_Mustache   0x0010
    "MouthOpen",  // AT_MouthOpen  0x0020
    "Expression", // AT_Expression 0x0040
    "Eye",        // AT_Eye        0x0080
    "BadImg",     // AT_BadImg     0x0100
    "Cropped",    // AT_Cropped    0x0200
    "Obscured",   // AT_Obscured   0x0400
    "BadEye",     // AT_BadEye     0x0800
    "Hat",        // AT_Hat        0x1000
    "Unused",     // AT_Hat        0x2000
    "MultiFace",  // AT_MultiFace  0x4000
    "BadTrain",   // AT_BadTrain   0x8000
};

static const unsigned AT_Male       = 0x0001; // gender, 1=male
static const unsigned AT_Child      = 0x0002; // child
static const unsigned AT_Glasses    = 0x0004; // face is wearing specs
static const unsigned AT_Beard      = 0x0008; // beard including possible mustache
static const unsigned AT_Mustache   = 0x0010; // mustache
static const unsigned AT_MouthOpen  = 0x0020; // mouth is open
static const unsigned AT_Expression = 0x0040; // non-neutral expression on face
static const unsigned AT_Eye        = 0x0080; // is eye landmark, used only during training

static const unsigned AT_BadImg     = 0x0100; // image is "bad" in some way (blurred, duplicated, etc.)
static const unsigned AT_Cropped    = 0x0200; // a landmark would be off the page
static const unsigned AT_Obscured   = 0x0400; // face is obscured e.g. by subject's hand
static const unsigned AT_BadEye     = 0x0800; // an eye is closed or obscured by hair etc.
static const unsigned AT_Hat        = 0x1000; // HAT desciptor (ignored if pyr lev > HAT_START_LEV)

// following are currently used only in aflw.shape
static const unsigned AT_MultiFace  = 0x4000; // multiple faces in the image
static const unsigned AT_BadTrain   = 0x8000; // face is not suitable for training face det
                                               // (by manual inspection of train face rects)

static const unsigned AT_Meta       = 0xFF000000; // "meta bits" used for face and pose detectors

static const unsigned AT_Pose       = 0x80000000; // 4 elements: yaw, pitch, roll, estimated_err

static const unsigned AT_EYAW00     = 0x81000000; // yaw00 face detector results, see DetPar
static const unsigned AT_EYAW22     = 0x82000000;
static const unsigned AT_EYAW45     = 0x83000000;
static const unsigned AT_EYAW_22    = 0x8A000000; // ms bit of nibble indicates negative yaw
static const unsigned AT_EYAW_45    = 0x8B000000;

static const unsigned AT_Any        = 0xFFFFFFFF; // special case, match any bit in Mask0

#endif // STASM_ATFACE_HPP
