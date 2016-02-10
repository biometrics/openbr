#include <opencv2/gpu/gpu.hpp>

using namespace cv;
using namespace cv::gpu;

namespace br { namespace cuda {
  void cudalbp_init_wrapper(uint8_t* lut, uint8_t** lutGpuPtrPtr);
  void cudalbp_wrapper(uint8_t* src, uint8_t* dst, uint8_t* lut, int imageWidth, int imageHeight, size_t step);
}}
