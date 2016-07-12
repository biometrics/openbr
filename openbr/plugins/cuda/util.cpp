#include <iostream>
#include <fstream>
using namespace std;

#include <Eigen/Dense>

namespace br {
  void dumpMatrixFloat(Eigen::MatrixXf& mat, char* filename) {
    ofstream myFile;
    myFile.open(filename);

    myFile << mat.rows() << " x " << mat.cols() << endl << endl;

    for (int i=0; i < mat.rows(); i++) {
      for (int j=0; j < mat.cols(); j++) {
        myFile << mat.data()[i*mat.cols()+j] << "\t";
      }
      myFile << endl;
    }

    myFile.close();
  }
  void dumpMatrixDouble(Eigen::MatrixXd& mat, char* filename) {
    ofstream myFile;
    myFile.open(filename);

    myFile << mat.rows() << " x " << mat.cols() << endl << endl;

    for (int i=0; i < mat.rows(); i++) {
      for (int j=0; j < mat.cols(); j++) {
        myFile << mat.data()[i*mat.cols()+j] << "\t";
      }
      myFile << endl;
    }

    myFile.close();
  }
  void dumpVectorFloat(Eigen::VectorXf& mat, char* filename) {
    ofstream myFile;
    myFile.open(filename);

    myFile << mat.rows() << endl << endl;

    for (int i=0; i < mat.rows(); i++) {
      myFile << mat.data()[i] << endl;
    }

    myFile.close();
  }
  void dumpVectorDouble(Eigen::VectorXd& mat, char* filename) {
    ofstream myFile;
    myFile.open(filename);

    myFile << mat.rows() << endl << endl;

    for (int i=0; i < mat.rows(); i++) {
      myFile << mat.data()[i] << endl;
    }

    myFile.close();
  }
}
