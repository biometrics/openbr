#include <pxcaccelerator.h>
#include <pxcface.h>
#include <pxcimage.h>
#include <pxcsession.h>

#include <openbr/plugins/openbr_internal.h>

using namespace br;

static PXCSession *pxcSession = NULL;
static PXCAccelerator *pxcAccelerator = NULL;

/*!
 * \ingroup initializers
 * \brief Initializes Intel Perceptual Computing SDK 2013
 * \author Josh Klontz \cite jklontz
 */
class IPC2013Initializer : public Initializer
{
	void initialize() const
	{
		PXCSession_Create(&pxcSession);
		pxcSession->CreateAccelerator(&pxcAccelerator);
	}
};

BR_REGISTER(Initializer, IPC2013Initializer)

/*!
 * \ingroup transforms
 * \brief Intel Perceptual Computing SDK 2013 Face Recognition
 * \author Josh Klontz \cite jklontz
 */
class IPC2013FaceRecognitionTransform : public UntrainableTransform
{
    Q_OBJECT
	
	void project(const Template &src, Template &dst) const
	{
		PXCImage::ImageInfo pxcImageInfo;
		pxcImageInfo.width = src.m().cols;
		pxcImageInfo.height = src.m().rows;
		pxcImageInfo.format = PXCImage::COLOR_FORMAT_RGB24;

		//PXCImage *pxcImage;
		//pxcAccelerator->CreateImage(&pxcImageInfo, 0, src.m().data, &pxcImage);
	}
};

BR_REGISTER(Transform, IPC2013FaceRecognitionTransfrom)

#include "ipc2013.moc"
