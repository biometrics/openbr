#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace cv;
using namespace cv::gpu;

namespace br { namespace cuda {
  void cudapca_initwrapper();

  void cudapca_loadwrapper(float* evPtr, int evRows, int evCols, float* meanPtr, int meanRows, int meanCols, int keep);
  void cudapca_trainwrapper();

  void cudapca_projectwrapper(float* src, float* dst);
}}
