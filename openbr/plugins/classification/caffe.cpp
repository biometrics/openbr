#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>
#include <openbr/core/qtutils.h>

#include <opencv2/imgproc/imgproc.hpp>
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
 * \brief A transform that wraps the Caffe deep learning library. This transform expects the input to a given Caffe model to be a MemoryDataLayer.
 * The output of the Caffe network is treated as a feature vector and is stored in dst. Batch processing is possible. For a given batch size set in
 * the memory data layer, src is expected to have an equal number of mats. Dst will always have the same size (number of mats) as src and the ordering
 * will be preserved, so dst[1] is the output of src[1] after it passes through the neural net.
 * \author Jordan Cheney \cite jcheney
 * \br_property QString model path to prototxt model file
 * \br_property QString weights path to caffemodel file
 * \br_property int gpuDevice ID of GPU to use. gpuDevice < 0 runs on the CPU only.
 * \br_link Caffe Integration Tutorial ../tutorials.md#caffe
 * \br_link Caffe website http://caffe.berkeleyvision.org
 */
class CaffeFVTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    Q_PROPERTY(QString model READ get_model WRITE set_model RESET reset_model STORED false)
    Q_PROPERTY(QString weights READ get_weights WRITE set_weights RESET reset_weights STORED false)
    Q_PROPERTY(int gpuDevice READ get_gpuDevice WRITE set_gpuDevice RESET reset_gpuDevice STORED false)
    BR_PROPERTY(QString, model, "")
    BR_PROPERTY(QString, weights, "")
    BR_PROPERTY(int, gpuDevice, -1)

    Resource<CaffeNet> caffeResource;

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

        MemoryDataLayer<float> *dataLayer = dynamic_cast<MemoryDataLayer<float> *>(net->layers()[0].get());

        if (src.size() != dataLayer->batch_size())
            qFatal("src should have %d (batch size) mats. It has %d mats.", dataLayer->batch_size(), src.size());

        dst.file = src.file;

        dataLayer->AddMatVector(src.toVector().toStdVector(), std::vector<int>(src.size(), 0));

        Blob<float> *output = net->ForwardPrefilled()[1]; // index 0 is the labels from the data layer (in this case the 0 array we passed in above).
                                                          // index 1 is the ouput of the final layer, which is what we want
        int dimFeatures = output->count() / dataLayer->batch_size();
        for (int n = 0; n < dataLayer->batch_size(); n++)
            dst += Mat(1, dimFeatures, CV_32FC1, output->mutable_cpu_data() + output->offset(n));

        caffeResource.release(net);
    }
};

BR_REGISTER(Transform, CaffeFVTransform)

/*!
 * \brief A transform that wraps the Caffe deep learning library. This transform expects the input to a given Caffe model to be a MemoryDataLayer.
 * The output of the Caffe network is treated as a classifier with one node per class. Dst is set equal to src and a list of labels and confidences
 * is stored in the metadata using the tags "Labels" and "Confidences". The size of the lists is equal to the batch size of the network. If the batch
 * size is 1 the lists are converted to an integer and floating point value respectively and are stored in metadata using "Label" and "Confidence" instead.
 * \author Jordan Cheney \cite jcheney
 * \br_property QString model path to prototxt model file
 * \br_property QString weights path to caffemodel file
 * \br_property int gpuDevice ID of GPU to use. gpuDevice < 0 runs on the CPU only.
 * \br_link Caffe Integration Tutorial ../tutorials.md#caffe
 * \br_link Caffe website http://caffe.berkeleyvision.org
 */
class CaffeClassifierTransform : public UntrainableMetaTransform
{
    Q_OBJECT

    Q_PROPERTY(QString model READ get_model WRITE set_model RESET reset_model STORED false)
    Q_PROPERTY(QString weights READ get_weights WRITE set_weights RESET reset_weights STORED false)
    Q_PROPERTY(int gpuDevice READ get_gpuDevice WRITE set_gpuDevice RESET reset_gpuDevice STORED false)
    BR_PROPERTY(QString, model, "")
    BR_PROPERTY(QString, weights, "")
    BR_PROPERTY(int, gpuDevice, -1)

    Resource<CaffeNet> caffeResource;

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
            qFatal("Integrating OpenBr with caffe requires the first layer in the network to be a MemoryDataLayer");

        MemoryDataLayer<float> *dataLayer = static_cast<MemoryDataLayer<float> *>(net->layers()[0].get());

        if (src.size() != dataLayer->batch_size())
            qFatal("src should have %d (batch size) mats. It has %d mats.", dataLayer->batch_size(), src.size());

        dst = src;

        dataLayer->AddMatVector(src.toVector().toStdVector(), std::vector<int>(src.size(), 0));

        Blob<float> *output = net->ForwardPrefilled()[1]; // index 0 is the labels from the data layer (in this case the 0 array we passed in above).
                                                          // index 1 is the ouput of the final layer, which is what we want

        QList<int> labels; QList<float> confidences;

        int dimFeatures = output->count() / dataLayer->batch_size();
        for (int n = 0; n < dataLayer->batch_size(); n++) {
            const float *data = output->cpu_data() + output->offset(n);

            int maxIdx = -1; float maxVal = -std::numeric_limits<float>::max();
            for (int d = 0; d < dimFeatures; d++) {
                if (data[d] > maxVal) {
                    maxVal = data[d];
                    maxIdx = d;
                }
            }

            labels.append(maxIdx);
            confidences.append(maxVal);
        }

        if (labels.size() == 1) {
            dst.file.set("Label", labels[0]);
            dst.file.set("Confidence", confidences[0]);
        } else {
            dst.file.setList<int>("Labels", labels);
            dst.file.setList<float>("Confidences", confidences);
        }

        caffeResource.release(net);
    }
};

BR_REGISTER(Transform, CaffeClassifierTransform)

} // namespace br

#include "classification/caffe.moc"
