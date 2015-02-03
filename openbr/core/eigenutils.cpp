#include "eigenutils.h"
#include <openbr/openbr_plugin.h>

using namespace Eigen;
using namespace cv;

void EigenUtils::printSize(Eigen::MatrixXf X) {
    qDebug() << "Rows=" << X.rows() << "\tCols=" << X.cols();
}

float EigenUtils::stddev(const Eigen::MatrixXf& x) {
    return sqrt((x.array() - x.mean()).pow(2).sum() / (x.cols() * x.rows()));
}

MatrixXf EigenUtils::removeRowCol(const MatrixXf X, int row, int col) {
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

MatrixXf EigenUtils::pointsToMatrix(const QList<QPointF> points, bool isAffine) {
    MatrixXf P(points.size(), isAffine ? 3 : 2);
    for (int i = 0; i < points.size(); i++) {
        P(i, 0) = points[i].x();
        P(i, 1) = points[i].y();
        if (isAffine)
            P(i, 2) = 1;
    }
    return P;
}

QList<QPointF> EigenUtils::matrixToPoints(const Eigen::MatrixXf P) {
    QList<QPointF> points;
    for (int i = 0; i < P.rows(); i++)
        points.append(QPointF(P(i, 0), P(i, 1)));
    return points;
}

//Converts x y points in a single vector to two column matrix
Eigen::MatrixXf EigenUtils::vectorToMatrix(const Eigen::MatrixXf vector) {
    int n = vector.rows();
    Eigen::MatrixXf matrix(n / 2, 2);
    for (int i = 0; i < n / 2; i++) {
        for (int j = 0; j < 2; j++) {
            matrix(i, j) = vector(i * 2 + j);
        }
    }
    return matrix;
}

Eigen::MatrixXf EigenUtils::matrixToVector(const Eigen::MatrixXf matrix) {
    int n2 = matrix.rows();
    Eigen::MatrixXf vector(n2 * 2, 1);
    for (int i = 0; i < n2; i++) {
        for (int j = 0; j < 2; j++) {
            vector(i * 2 + j) = matrix(i, j);
        }
    }
    return vector;
}
