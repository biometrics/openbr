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

float eigMean(const Eigen::MatrixXf& x) {
    return x.array().sum() / (x.rows() * x.cols());
}

float eigStd(const Eigen::MatrixXf& x) {
    float mean = eigMean(x);
    return sqrt((x.array() - mean).pow(2).sum() / (x.cols() * x.rows()));
}

MatrixXf removeRowCol(const MatrixXf X, int row, int col) {
    MatrixXf Y(X.rows() - 1,X.cols() - 1);

    for (int i1 = 0, i2 = 0; i1 < X.rows(); i1++) {
        if (i1 == row)
            continue;
        i2++;

        for (int j1 = 0, j2 = 0; j1 < X.cols(); j1++) {
            if (j1 == col)
                continue;
            j2++;

            Y(i2,j2) = X(i1,j1);
        }
    }
    return Y;
}

MatrixXf pointsToMatrix(const QList<QPointF> points, bool isAffine) {
    MatrixXf P(points.size(), isAffine ? 3 : 2);
    for (int i = 0; i < points.size(); i++) {
        P(i, 0) = points[i].x();
        P(i, 1) = points[i].y();
        if (isAffine)
            P(i, 2) = 1;
    }
    return P;
}

QList<QPointF> matrixToPoints(const Eigen::MatrixXf P) {
    QList<QPointF> points;
    for (int i = 0; i < P.rows(); i++)
        points.append(QPointF(P(i, 0), P(i, 1)));
    return points;
}

//Converts x y points in a single vector to two column matrix
Eigen::MatrixXf vectorToMatrix(const Eigen::MatrixXf vector) {
    int n = vector.rows();
    Eigen::MatrixXf matrix(n / 2, 2);
    for (int i = 0; i < n / 2; i++) {
        for (int j = 0; j < 2; j++) {
            matrix(i, j) = vector(i * 2 + j);
        }
    }
    return matrix;
}

Eigen::MatrixXf matrixToVector(const Eigen::MatrixXf matrix) {
    int n2 = matrix.rows();
    Eigen::MatrixXf vector(n2 * 2, 1);
    for (int i = 0; i < n2; i++) {
        for (int j = 0; j < 2; j++) {
            vector(i * 2 + j) = matrix(i, j);
        }
    }
    return vector;
}

Eigen::MatrixXf toEigen(const Mat m) {
    if (m.type() != CV_32F)
        qFatal("Mat to Eigen Converstation only supports CV_32F");

    Eigen::MatrixXf data(m.rows, m.cols);
    return Eigen::Map<const Eigen::MatrixXf>(m.ptr<float>(), m.rows, m.cols);
}

