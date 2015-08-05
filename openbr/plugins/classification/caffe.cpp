#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

#include <caffe/caffe.hpp>

using caffe::Caffe;
using caffe::Net;
using caffe::MemoryDataLayer;
using caffe::Blob;
using caffe::shared_ptr;

using namespace cv;

namespace br
{

// Net doesn't expose a default constructor which is expected by the default resource allocator.
// To get around that we make this custom stub class which has a default constructor that passes
// empty values to the Net constructor.
class CaffeNet : public Net<float>
{
public:
    CaffeNet() : Net<float>("", caffe::TEST) {}
    CaffeNet(const QString &model, caffe::Phase phase) : Net<float>(model.toStdString(), phase) {}
};

class CaffeResourceMaker : public ResourceMaker<CaffeNet>
{
    QString model;
    QString weights;
    int gpuDevice;

public:
    CaffeResourceMaker(const QString &model, const QString &weights, int gpuDevice) : model(model), weights(weights), gpuDevice(gpuDevice) {}

private:
    CaffeNet *make() const
    {
        if (gpuDevice >= 0) {
            Caffe::SetDevice(gpuDevice);
            Caffe::set_mode(Caffe::GPU);
        } else {
            Caffe::set_mode(Caffe::CPU);
        }

        CaffeNet *net = new CaffeNet(model, caffe::TEST);
        net->CopyTrainedLayersFrom(weights.toStdString());
        return net;
    }
};

/*!
 * \brief The base transform for wrapping the Caffe deep learning library. This transform expects the input to a given Caffe model to be a MemoryDataLayer.
 * The output of the forward pass of the Caffe network is stored in dst as a list of matrices, the size of which is equal to the batch_size of the network.
 * Children of this transform should process dst to acheieve specifc use cases.
 * \author Jordan Cheney \cite JordanCheney
 * \br_property QString model path to prototxt model file
 * \br_property QString weights path to caffemodel file
 * \br_property int gpuDevice ID of GPU to use. gpuDevice < 0 runs on the CPU only.
 * \br_link Caffe Integration Tutorial ../tutorials.md#caffe
 * \br_link Caffe website http://caffe.berkeleyvision.org
 */
class CaffeBaseTransform : public UntrainableMetaTransform
{
    Q_OBJECT

public:
    Q_PROPERTY(QString model READ get_model WRITE set_model RESET reset_model STORED false)
    Q_PROPERTY(QString weights READ get_weights WRITE set_weights RESET reset_weights STORED false)
    Q_PROPERTY(int gpuDevice READ get_gpuDevice WRITE set_gpuDevice RESET reset_gpuDevice STORED false)
    BR_PROPERTY(QString, model, "")
    BR_PROPERTY(QString, weights, "")
    BR_PROPERTY(int, gpuDevice, -1)

    Resource<CaffeNet> caffeResource;

protected:
    void init()
    {
        caffeResource.setResourceMaker(new CaffeResourceMaker(model, weights, gpuDevice));
    }

    bool timeVarying() const
    {
        return gpuDevice < 0 ? false : true;
    }

    void project(const Template &src, Template &dst) const
    {
        CaffeNet *net = caffeResource.acquire();

        if (net->layers()[0]->layer_param().type() != "MemoryData")
            qFatal("OpenBR requires the first layer in the network to be a MemoryDataLayer");

        MemoryDataLayer<float> *dataLayer = static_cast<MemoryDataLayer<float> *>(net->layers()[0].get());

        if (src.size() != dataLayer->batch_size())
            qFatal("src should have %d (batch size) mats. It has %d mats.", dataLayer->batch_size(), src.size());

        dataLayer->AddMatVector(src.toVector().toStdVector(), std::vector<int>(src.size(), 0));

        net->ForwardPrefilled();
        Blob<float> *output = net->blobs().back().get();

        int dimFeatures = output->count() / dataLayer->batch_size();
        for (int n = 0; n < dataLayer->batch_size(); n++)
            dst += Mat(1, dimFeatures, CV_32FC1, output->mutable_cpu_data() + output->offset(n));

        caffeResource.release(net);
    }
};

/*!
 * \brief This transform treats the output of the network as a feature vector and appends it unchanged to dst. Dst will have
 * length equal to the batch size of the network.
 * \author Jordan Cheney \cite JordanCheney
 * \br_property QString model path to prototxt model file
 * \br_property QString weights path to caffemodel file
 * \br_property int gpuDevice ID of GPU to use. gpuDevice < 0 runs on the CPU only.
 */
class CaffeFVTransform : public CaffeBaseTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        Template caffeOutput;
        CaffeBaseTransform::project(src, caffeOutput);

        dst.file = src.file;
        dst.append(caffeOutput);
    }
};

BR_REGISTER(Transform, CaffeFVTransform)

/*!
 * \brief This transform treats the output of the network as a score distribution for an arbitrary number of classes.
 * The maximum score and location for each input image is determined and stored in the template metadata. The template
 * matrix is not changed. If the network batch size is > 1, the results are stored as lists in the dst template's metadata
 * using the keys "Labels" and "Confidences" respectively. The length of these lists is equivalent to the provided batch size.
 * If batch size == 1, the results are stored as a float and int using the keys "Label", and "Confidence" respectively.
 * \author Jordan Cheney \cite jcheney
 * \br_property QString model path to prototxt model file
 * \br_property QString weights path to caffemodel file
 * \br_property int gpuDevice ID of GPU to use. gpuDevice < 0 runs on the CPU only.
 */
class CaffeClassifierTransform : public CaffeBaseTransform
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        Template caffeOutput;
        CaffeBaseTransform::project(src, caffeOutput);

        dst = src;

        QList<int> labels; QList<float> confidences;

        foreach (const Mat &m, caffeOutput) {
            double maxVal; int maxLoc;
            minMaxIdx(m, NULL, &maxVal, NULL, &maxLoc);

            labels.append(maxLoc);
            confidences.append(maxVal);
        }

        if (labels.size() == 1) {
            dst.file.set("Label", labels[0]);
            dst.file.set("Confidence", confidences[0]);
        } else {
            dst.file.setList<int>("Labels", labels);
            dst.file.setList<float>("Confidences", confidences);
        }
    }
};

BR_REGISTER(Transform, CaffeClassifierTransform)

} // namespace br

#include "classification/caffe.moc"
