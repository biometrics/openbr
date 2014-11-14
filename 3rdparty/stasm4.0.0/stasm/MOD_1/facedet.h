// facedet.h: find faces in images (frontal model version)
//
// Copyright (C) 2005-2013, Stephen Milborrow

#ifndef STASM_FACEDET_H
#define STASM_FACEDET_H

#include <opencv2/objdetect/objdetect.hpp>

namespace stasm
{
class FaceDet
{
public:
    void OpenFaceDetector_( // called by stasm_init, init face det from XML file
        const char* datadir,      // in: directory of face detector files
        void*       detparams);   // in: unused (func signature compatibility)

    // Call DetectFaces_ once per image.  Then call NextFace_ repeatedly to get
    // all the faces in the image, one by one.  When there are no more faces in
    // the image, NextFace_ will return detpar.x set to INVALID.
    //
    // Note: This version of NextFace_ always sets detpar.rot to INVALID.
    // Thus the calling routine (StartShapeAndRoi1) must estimate the
    // rotation from the intereye angle.  However, if an alternative
    // implementation were to initialize detpar.rot, StartShapeAndRoi1 would
    // use it (instead of estimating it from the eye angle).

    void DetectFaces_(            // call once per image to find all the faces
        const Image& img,         // in: the image (grayscale)
        const char*  imgpath,     // in: unused (match virt func signature)
        bool         multiface,   // in: if false, want only the best face
        int          minwidth,    // in: min face width as percent of img width
        void*        user,
        cv::CascadeClassifier cascade);       // in: unused (match virt func signature)

    const DetPar NextFace_(void); // get next face from faces found by DetectFaces_

    FaceDet() {}                  // constructor


private:
    vector<DetPar>  detpars_;     // all the valid faces in the current image

    int             iface_;       // index of current face for NextFace_
                                  // indexes into detpars_

    DISALLOW_COPY_AND_ASSIGN(FaceDet);

}; // end class FaceDet

} // namespace stasm
#endif // STASM_FACEDET.H
