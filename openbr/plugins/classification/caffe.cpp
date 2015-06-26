#include <openbr/plugins/openbr_internal.h>

#include <caffe/caffe.hpp>

using caffe::Caffe;
using caffe::Solver;
using caffe::Net;
using caffe::Blob;
using caffe::shared_ptr;
using caffe::vector;

namespace br
{

/*!
 * \brief A transform that wraps the Caffe Deep learning library
 * \author Jordan Cheney \cite JordanCheney
 * \
 */
class CaffeTransform : public Transform
{
    Q_OBJECT

    Q_PROPERTY(QString modelFile READ get_modelFile WRITE set_modelFile RESET reset_modelFile STORED false)
    Q_PROPERTY(QString solverFile READ get_solverFile WRITE set_solverFile RESET reset_solverFile STORED false)
    Q_PROPERTY(QString weightsFile READ get_weightsFile WRITE set_weightsFile RESET reset_weightsFile STORED false)
    Q_PROPERTY(int gpuDevice READ get_gpuDevice WRITE set_gpuDevice RESET reset_gpuDevice STORED false)
    BR_PROPERTY(QString, modelFile, "")
    BR_PROPERTY(QString, solverFile, "")
    BR_PROPERTY(QString, weightsFile, "")
    BR_PROPERTY(int, gpuDevice, -1)

    void init()
    {
        if (gpuDevice >= 0) {
            Caffe::SetDevice(gpuDevice);
            Caffe::set_mode(Caffe::GPU);
        } else {
            Caffe::set_mode(Caffe::CPU);
        }
    }

    void train(const TemplateList &data)
    {
        (void) data;

        caffe::SolverParameter solver_param;
        caffe::ReadProtoFromTextFileOrDie(solverFile.toStdString(), &solver_param);

        shared_ptr<Solver<float> > solver(caffe::GetSolver<float>(solver_param));
        solver->Solve();
    }

    void project(const Template &src, Template &dst) const
    {
        (void)src; (void)dst;
        Net<float> net(modelFile.toStdString(), caffe::TEST);
        net.CopyTrainedLayersFrom(weightsFile.toStdString());

        vector<Blob<float> *> bottom_vec; // perhaps src data should go here?
        vector<int> test_score_output_id;
        vector<float> test_score;

        float loss;
        const vector<Blob<float> *> &result = net.Forward(bottom_vec, &loss);

        int idx = 0;
        for (int i = 0; i < (int)result.size(); i++) {
            const float *result_data = result[i]->cpu_data();
            for (int j = 0; j < result[i]->count(); j++, idx++) {
                test_score.push_back(result_data[j]);
                test_score_output_id.push_back(i);

                if (Globals->verbose)
                    qDebug("%s = %f", net.blob_names()[net.output_blob_indices()[i]].c_str(), result_data[j]);
            }
        }
    }
};

BR_REGISTER(Transform, CaffeTransform)

} // namespace br

#include "classification/caffe.moc"
