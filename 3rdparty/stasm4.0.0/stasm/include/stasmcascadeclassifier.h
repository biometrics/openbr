#ifndef STASMCASCADECLASSIFIER_H
#define STASMCASCADECLASSIFIER_H

#include <opencv2/objdetect/objdetect.hpp>
#include <QString>

class StasmCascadeClassifier
{

public:
    cv::CascadeClassifier faceCascade;
    cv::CascadeClassifier mouthCascade;
    cv::CascadeClassifier leftEyeCascade;
    cv::CascadeClassifier rightEyeCascade;

    bool load(const std::string &path) {
        if (!faceCascade.load(path + "haarcascades/haarcascade_frontalface_alt2.xml")   ||
            !mouthCascade.load(path + "haarcascades/haarcascade_mcs_mouth.xml")  ||
            !leftEyeCascade.load(path + "haarcascades/haarcascade_mcs_lefteye.xml")     ||
            !rightEyeCascade.load(path + "haarcascades/haarcascade_mcs_righteye.xml"))  return false;

        return true;
    }

};

#endif // STASMCASCADECLASSIFIER_H
