/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2016 Greg Shrock, Colin Heinzmann                               *
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License");           *
 * you may not use this file except in compliance with the License.          *
 * You may obtain a copy of the License at                                   *
 *                                                                           *
 *     http://www.apache.org/licenses/LICENSE-2.0                            *
 *                                                                           *
 * Unless required by applicable law or agreed to in writing, software       *
 * distributed under the License is distributed on an "AS IS" BASIS,         *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 * See the License for the specific language governing permissions and       *
 * limitations under the License.                                            *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */



#include <iostream>
using namespace std;

#include <sys/types.h>
#include <unistd.h>

#include <pthread.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <limits>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

using namespace cv;

// definitions from the CUDA source file
namespace br { namespace cuda { namespace affine {
	void resizeWrapper(void* srcPtr, void** dstPtr, int src_rows, int src_cols, int dst_rows, int dst_cols);
	void wrapper(void* srcPtr, void** dstPtr, Mat affineTransform, int src_rows, int src_cols, int dst_rows, int dst_cols);
}}}

namespace br
{

	/*!
	* \ingroup transforms
	* \brief Performs a two or three point registration on the GPU.  Modified from stock OpenBR implementation.  Only supports single-point input bilinear transformation.
	* \author Greg Schrock \cite gls022
  * \author Colin Heinzmann \cite DepthDeluxe
	* \note Method: Area should be used for shrinking an image, Cubic for slow but accurate enlargment, Bilin for fast enlargement.
	*/
	class CUDAAffineTransform : public UntrainableTransform
	{
	    Q_OBJECT

  private:
	    Q_PROPERTY(int width READ get_width WRITE set_width RESET reset_width STORED false)
	    Q_PROPERTY(int height READ get_height WRITE set_height RESET reset_height STORED false)
	    Q_PROPERTY(float x1 READ get_x1 WRITE set_x1 RESET reset_x1 STORED false)
	    Q_PROPERTY(float y1 READ get_y1 WRITE set_y1 RESET reset_y1 STORED false)
	    BR_PROPERTY(int, width, 64)
	    BR_PROPERTY(int, height, 64)
	    BR_PROPERTY(float, x1, 0)
	    BR_PROPERTY(float, y1, 0)

	    static Point2f getThirdAffinePoint(const Point2f &a, const Point2f &b)
	    {
	        float dx = b.x - a.x;
	        float dy = b.y - a.y;
	        return Point2f(a.x - dy, a.y + dx);
	    }

	    void project(const Template &src, Template &dst) const
	    {
	        Point2f dstPoints[3];
	        dstPoints[0] = Point2f(x1*width, y1*height);
          dstPoints[1] = Point2f((1-x1)*width, (1-y1)*height);
          dstPoints[2] = getThirdAffinePoint(dstPoints[0], dstPoints[1]);

	        Point2f srcPoints[3];
	        if (src.file.contains("Affine_0") &&
	            src.file.contains("Affine_1") &&
	            src.file.contains("Affine_2")) {
	            srcPoints[0] = OpenCVUtils::toPoint(src.file.get<QPointF>("Affine_0"));
	            srcPoints[1] = OpenCVUtils::toPoint(src.file.get<QPointF>("Affine_1"));
	        } else {
	            const QList<Point2f> landmarks = OpenCVUtils::toPoints(src.file.points());

	            if (landmarks.size() < 2) {
                  void* const* srcDataPtr = src.m().ptr<void*>();
                  int rows = *((int*)srcDataPtr[1]);
                  int cols = *((int*)srcDataPtr[2]);
                  int type = *((int*)srcDataPtr[3]);

                  if (type != CV_8UC1) {
                    cout << "ERR: Invalid image format!" << endl;
                    return;
                  }

                  Mat dstMat = Mat(src.m().rows, src.m().cols, src.m().type());
                  void** dstDataPtr = dstMat.ptr<void*>();

                  dstDataPtr[1] = srcDataPtr[1]; *((int*)dstDataPtr[1]) = height;  // rows
                  dstDataPtr[2] = srcDataPtr[2]; *((int*)dstDataPtr[2]) = width;   // cols
                  dstDataPtr[3] = srcDataPtr[3];

                  cuda::affine::resizeWrapper(srcDataPtr[0], &dstDataPtr[0], rows, cols, height, width);
                  dst = dstMat;
	                return;
	            } else {
	                srcPoints[0] = landmarks[0];
	                srcPoints[1] = landmarks[1];
	            }
	        }
	        srcPoints[2] = getThirdAffinePoint(srcPoints[0], srcPoints[1]);

	        Mat affineTransform = getAffineTransform(srcPoints, dstPoints);

	        void* const* srcDataPtr = src.m().ptr<void*>();
	        int rows = *((int*)srcDataPtr[1]);
	        int cols = *((int*)srcDataPtr[2]);
	        int type = *((int*)srcDataPtr[3]);

          if (type != CV_8UC1) {
            cout << "ERR: Invalid image format!" << endl;
            return;
          }


	        Mat dstMat = Mat(src.m().rows, src.m().cols, src.m().type());
	        void** dstDataPtr = dstMat.ptr<void*>();

	        dstDataPtr[1] = srcDataPtr[1]; *((int*)dstDataPtr[1]) = height;  // rows
	        dstDataPtr[2] = srcDataPtr[2]; *((int*)dstDataPtr[2]) = width;   // cols
	        dstDataPtr[3] = srcDataPtr[3];

	        cuda::affine::wrapper(srcDataPtr[0], &dstDataPtr[0], affineTransform, rows, cols, height, width);

	        dst = dstMat;
	    }
	};

	BR_REGISTER(Transform, CUDAAffineTransform)

} // namespace br

#include "cuda/cudaaffine.moc"
