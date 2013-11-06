#include "eigenutils.h"
#include <openbr/openbr_plugin.h>

using namespace Eigen;
using namespace cv;

//Helper function to quickly write eigen matrix to disk. Not efficient.
void writeEigen(MatrixXf X, QString filename) {
    Mat m(X.rows(),X.cols(),CV_32FC1);
    for (int i = 0; i < X.rows(); i++) {
        for (int j = 0; j < X.cols(); j++) {
            m.at<float>(i,j) = X(i,j);
        }
    }
    QScopedPointer<br::Format> format(br::Factory<br::Format>::make(filename));
    format->write(br::Template(m));
}


