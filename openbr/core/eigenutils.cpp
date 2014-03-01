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

void writeEigen(MatrixXd X, QString filename) {
    Mat m(X.rows(),X.cols(),CV_32FC1);
    for (int i = 0; i < X.rows(); i++) {
        for (int j = 0; j < X.cols(); j++) {
            m.at<float>(i,j) = (float)X(i,j);
        }
    }
    QScopedPointer<br::Format> format(br::Factory<br::Format>::make(filename));
    format->write(br::Template(m));
}

void writeEigen(VectorXd X, QString filename) {
    Mat m(X.size(),1,CV_32FC1);
    for (int i = 0; i < X.rows(); i++) {
        m.at<float>(i,0) = (float)X(i);
    }
    QScopedPointer<br::Format> format(br::Factory<br::Format>::make(filename));
    format->write(br::Template(m));
}

void writeEigen(VectorXf X, QString filename) {
    Mat m(X.size(),1,CV_32FC1);
    for (int i = 0; i < X.rows(); i++) {
        m.at<float>(i,0) = X(i);
    }
    QScopedPointer<br::Format> format(br::Factory<br::Format>::make(filename));
    format->write(br::Template(m));
}

void printEigen(Eigen::MatrixXd X) {
    for (int i = 0; i < X.rows(); i++) {
        QString str;
        for (int j = 0; j < X.cols(); j++) {
            str.append(QString::number(X(i,j)) + " ");
        }
        qDebug() << str;
    }
}
void printEigen(Eigen::MatrixXf X) {
    for (int i = 0; i < X.rows(); i++) {
        QString str;
        for (int j = 0; j < X.cols(); j++) {
            str.append(QString::number(X(i,j)) + " ");
        }
        qDebug() << str;
    }
}

void printSize(Eigen::MatrixXf X) {
    qDebug() << "Rows=" << X.rows() << "\tCols=" << X.cols();
}
