/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2012 The MITRE Corporation                                      *
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

#include <Eigen/Dense>
#include <openbr/plugins/openbr_internal.h>

#include <openbr/core/common.h>
#include <openbr/core/eigenutils.h>
#include <openbr/core/opencvutils.h>

namespace br
{

/*!
 * \ingroup initializers
 * \brief Initialize Eigen
 * \br_link http://eigen.tuxfamily.org/dox/TopicMultiThreading.html
 * \author Scott Klum \cite sklum
 */
class EigenInitializer : public Initializer
{
    Q_OBJECT

    void initialize() const
    {
        Eigen::initParallel();
    }
};

BR_REGISTER(Initializer, EigenInitializer)

/*!
 * \ingroup transforms
 * \brief Projects input into learned Principal Component Analysis subspace.
 * \author Brendan Klare \cite bklare
 * \author Josh Klontz \cite jklontz
 *
 * \br_property float keep Options are: [keep < 0 - All eigenvalues are retained, keep == 0 - No PCA is performed and the eigenvectors form an identity matrix, 0 < keep < 1 - Keep is the fraction of the variance to retain, keep >= 1 - keep is the number of leading eigenvectors to retain] Default is 0.95.
 * \br_property int drop The number of leading eigen-dimensions to drop.
 * \br_property bool whiten Whether or not to perform PCA whitening (i.e., normalize variance of each dimension to unit norm)
 */
class PCATransform : public Transform
{
    Q_OBJECT
    friend class LDATransform;

protected:
    Q_PROPERTY(float keep READ get_keep WRITE set_keep RESET reset_keep STORED false)
    Q_PROPERTY(int drop READ get_drop WRITE set_drop RESET reset_drop STORED false)
    Q_PROPERTY(bool whiten READ get_whiten WRITE set_whiten RESET reset_whiten STORED false)

    BR_PROPERTY(float, keep, 0.95)
    BR_PROPERTY(int, drop, 0)
    BR_PROPERTY(bool, whiten, false)

    Eigen::VectorXf mean, eVals;
    Eigen::MatrixXf eVecs;

    int originalRows;

public:
    PCATransform() : keep(0.95), drop(0), whiten(false) {}

private:
    double residualReconstructionError(const Template &src) const
    {
        Template proj;
        project(src, proj);

        Eigen::Map<const Eigen::VectorXf> srcMap(src.m().ptr<float>(), src.m().rows*src.m().cols);
        Eigen::Map<Eigen::VectorXf> projMap(proj.m().ptr<float>(), keep);

        return (srcMap - mean).squaredNorm() - projMap.squaredNorm();
    }

    void train(const TemplateList &trainingSet)
    {
        if (trainingSet.first().m().type() != CV_32FC1)
            qFatal("Requires single channel 32-bit floating point matrices.");

        originalRows = trainingSet.first().m().rows;
        int dimsIn = trainingSet.first().m().rows * trainingSet.first().m().cols;
        const int instances = trainingSet.size();

        // Map into 64-bit Eigen matrix
        Eigen::MatrixXd data(dimsIn, instances);
        for (int i=0; i<instances; i++)
            data.col(i) = Eigen::Map<const Eigen::MatrixXf>(trainingSet[i].m().ptr<float>(), dimsIn, 1).cast<double>();

        trainCore(data);
    }

    void project(const Template &src, Template &dst) const
    {
        dst = cv::Mat(1, keep, CV_32FC1);

        // Map Eigen into OpenCV
        Eigen::Map<const Eigen::MatrixXf> inMap(src.m().ptr<float>(), src.m().rows*src.m().cols, 1);
        Eigen::Map<Eigen::MatrixXf> outMap(dst.m().ptr<float>(), keep, 1);

        // Do projection
        outMap = eVecs.transpose() * (inMap - mean);
    }

    void store(QDataStream &stream) const
    {
        stream << keep << drop << whiten << originalRows << mean << eVals << eVecs;
    }

    void load(QDataStream &stream)
    {
        stream >> keep >> drop >> whiten >> originalRows >> mean >> eVals >> eVecs;
    }

protected:
    void trainCore(Eigen::MatrixXd data)
    {
        int dimsIn = data.rows();
        int instances = data.cols();
        const bool dominantEigenEstimation = (dimsIn > instances);

        Eigen::MatrixXd allEVals, allEVecs;
        if (keep != 0) {
            // Compute and remove mean
            mean = Eigen::VectorXf(dimsIn);
            for (int i=0; i<dimsIn; i++) mean(i) = data.row(i).sum() / (float)instances;
            for (int i=0; i<dimsIn; i++) data.row(i).array() -= mean(i);

            // Calculate covariance matrix
            Eigen::MatrixXd cov;
            if (dominantEigenEstimation) cov = data.transpose() * data / (instances-1.0);
            else                         cov = data * data.transpose() / (instances-1.0);

            // Compute eigendecomposition. Returns eigenvectors/eigenvalues in increasing order by eigenvalue.
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eSolver(cov);
            allEVals = eSolver.eigenvalues();
            allEVecs = eSolver.eigenvectors();
            if (dominantEigenEstimation) allEVecs = data * allEVecs;
        } else {
            // Null case
            mean = Eigen::VectorXf::Zero(dimsIn);
            allEVecs = Eigen::MatrixXd::Identity(dimsIn, dimsIn);
            allEVals = Eigen::VectorXd::Ones(dimsIn);
        }

        if (keep <= 0) {
            keep = dimsIn - drop;
        } else if (keep < 1) {
            // Keep eigenvectors that retain a certain energy percentage.
            const double totalEnergy = allEVals.sum();
            if (totalEnergy == 0) {
                keep = 0;
            } else {
                double currentEnergy = 0;
                int i=0;
                while ((currentEnergy / totalEnergy < keep) && (i < allEVals.rows())) {
                    currentEnergy += allEVals(allEVals.rows()-(i+1));
                    i++;
                }
                keep = i - drop;
            }
        } else {
            if (keep + drop > allEVals.rows()) {
                qWarning("Insufficient samples, needed at least %d but only got %d.", (int)keep + drop, (int)allEVals.rows());
                keep = allEVals.rows() - drop;
            }
        }

        // Keep highest energy vectors
        eVals = Eigen::VectorXf((int)keep, 1);
        eVecs = Eigen::MatrixXf(allEVecs.rows(), (int)keep);
        for (int i=0; i<keep; i++) {
            int index = allEVals.rows()-(i+drop+1);
            eVals(i) = allEVals(index);
            eVecs.col(i) = allEVecs.col(index).cast<float>() / allEVecs.col(index).norm();
            if (whiten) eVecs.col(i) /= sqrt(eVals(i));
        }

        // Debug output
        if (Globals->verbose) qDebug() << "PCA Training:\n\tDimsIn =" << dimsIn << "\n\tKeep =" << keep;
    }

    void writeEigenVectors(const Eigen::MatrixXd &allEVals, const Eigen::MatrixXd &allEVecs) const
    {
        const int originalCols = mean.rows() / originalRows;

        { // Write out mean image
            cv::Mat out(originalRows, originalCols, CV_32FC1);
            Eigen::Map<Eigen::MatrixXf> outMap(out.ptr<float>(), mean.rows(), 1);
            outMap = mean.col(0);
            // OpenCVUtils::saveImage(out, Globals->Debug+"/PCA/eigenVectors/mean.png");
        }

        // Write out sample eigen vectors (16 highest, 8 lowest), filename = eigenvalue.
        for (int k=0; k<(int)allEVals.size(); k++) {
            if ((k < 8) || (k >= (int)allEVals.size()-16)) {
                cv::Mat out(originalRows, originalCols, CV_64FC1);
                Eigen::Map<Eigen::MatrixXd> outMap(out.ptr<double>(), mean.rows(), 1);
                outMap = allEVecs.col(k);
                // OpenCVUtils::saveImage(out, Globals->Debug+"/PCA/eigenVectors/"+QString::number(allEVals(k),'f',0)+".png");
            }
        }
    }
};

BR_REGISTER(Transform, PCATransform)

/*!
 * \ingroup transforms
 * \brief Projects input into learned Linear Discriminant Analysis subspace.
 * \author Brendan Klare \cite bklare
 * \author Josh Klontz \cite jklontz
 * \br_property float pcaKeep If <= 1, the percentage of variance to retain in initial PCA step. If > 1, the number of dimensions to keep.
 * \br_property bool pcaWhiten Whether or not to perform whitening during PCA step
 * \br_property int directLDA Whether or not to use the Direct LDA algorithm.
 * \br_property float directDrop Parameter for Direct LDA to specify how many leading vectors in within class matrix to drop. Based on variance.
 * \br_property QString inputVariable Metadata key for subject labels. 
 * \br_property bool isBinary Whether or not to perform binary LDA. Default is multi-class LDA (i.e., distance metric learning).
 * \br_property bool normalize For binary LDA, whether or not to z-score normalize projection.
 */
class LDATransform : public Transform
{
    friend class SparseLDATransform;

    Q_OBJECT
    Q_PROPERTY(float pcaKeep READ get_pcaKeep WRITE set_pcaKeep RESET reset_pcaKeep STORED false)
    Q_PROPERTY(bool pcaWhiten READ get_pcaWhiten WRITE set_pcaWhiten RESET reset_pcaWhiten STORED false)
    Q_PROPERTY(int directLDA READ get_directLDA WRITE set_directLDA RESET reset_directLDA STORED false)
    Q_PROPERTY(float directDrop READ get_directDrop WRITE set_directDrop RESET reset_directDrop STORED false)
    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED false)
    Q_PROPERTY(bool isBinary READ get_isBinary WRITE set_isBinary RESET reset_isBinary STORED false)
    Q_PROPERTY(bool normalize READ get_normalize WRITE set_normalize RESET reset_normalize STORED false)
    BR_PROPERTY(float, pcaKeep, 0.98)
    BR_PROPERTY(bool, pcaWhiten, false)
    BR_PROPERTY(int, directLDA, 0)
    BR_PROPERTY(float, directDrop, 0.1)
    BR_PROPERTY(QString, inputVariable, "Label")
    BR_PROPERTY(bool, isBinary, false)
    BR_PROPERTY(bool, normalize, true)

    int dimsOut;
    Eigen::VectorXf mean;
    Eigen::MatrixXf projection;
    float stdDev;

    void train(const TemplateList &_trainingSet)
    {
        // creates "Label"
        TemplateList trainingSet = TemplateList::relabel(_trainingSet, inputVariable, isBinary);
        int instances = trainingSet.size();

        // Perform PCA dimensionality reduction
        PCATransform pca;
        pca.keep = pcaKeep;
        pca.whiten = pcaWhiten;
        pca.train(trainingSet);
        mean = pca.mean;

        TemplateList ldaTrainingSet;
        static_cast<Transform*>(&pca)->project(trainingSet, ldaTrainingSet);

        int dimsIn = ldaTrainingSet.first().m().rows * ldaTrainingSet.first().m().cols;

        // OpenBR ensures that class values range from 0 to numClasses-1.
        // Label exists because we created it earlier with relabel
        QList<int> classes = File::get<int>(trainingSet, "Label");
        QMap<int, int> classCounts = trainingSet.countValues<int>("Label");
        const int numClasses = classCounts.size();

        // Map Eigen into OpenCV
        Eigen::MatrixXd data = Eigen::MatrixXd(dimsIn, instances);
        for (int i=0; i<instances; i++)
            data.col(i) = Eigen::Map<const Eigen::MatrixXf>(ldaTrainingSet[i].m().ptr<float>(), dimsIn, 1).cast<double>();

        // Removing class means
        Eigen::MatrixXd classMeans = Eigen::MatrixXd::Zero(dimsIn, numClasses);
        for (int i=0; i<instances; i++)  classMeans.col(classes[i]) += data.col(i);
        for (int i=0; i<numClasses; i++) classMeans.col(i) /= classCounts[i];
        for (int i=0; i<instances; i++)  data.col(i) -= classMeans.col(classes[i]);

        PCATransform space1;

        if (!directLDA)
        {
            // The number of LDA dimensions is limited by the degrees
            // of freedom of scatter matrix computed from 'data'. Because
            // the mean of each class is removed (lowering degree of freedom
            // one per class), the total rank of the covariance/scatter
            // matrix that will be computed in PCA is bound by instances - numClasses.
            space1.keep = std::min(dimsIn, instances-numClasses);
            space1.trainCore(data);

            // Divide each eigenvector by sqrt of eigenvalue.
            // This has the effect of whitening the within-class scatter.
            // In effect, this minimizes the within-class variation energy.
            for (int i=0; i<space1.keep; i++) space1.eVecs.col(i) /= pow((double)space1.eVals(i),0.5);
        }
        else if (directLDA == 2)
        {
            space1.drop = instances - numClasses;
            space1.keep = std::min(dimsIn, instances) - space1.drop;
            space1.trainCore(data);
        }
        else
        {
            // Perform (modified version of) Direct LDA

            // Direct LDA uses to the Null space of the within-class scatter.
            // Thus, the lower rank, is used to our benefit. We are not discarding
            // these vectors now (in non-direct code we use the keep parameter
            // to discard Null space). We keep the Null space b/c this is where
            // the within-class scatter goes to zero, i.e. it is very useful.
            space1.keep = dimsIn;
            space1.trainCore(data);

            if (dimsIn > instances - numClasses) {
                // Here, we are replacing the eigenvalue of the  null space
                // eigenvectors with the eigenvalue (divided by 2) of the
                // smallest eigenvector from the row space eigenvector.
                // This allows us to scale these null-space vectors (otherwise
                // it is a divide by zero.
                double null_eig = space1.eVals(instances - numClasses - 1) / 2;
                for (int i = instances - numClasses; i < dimsIn; i++)
                    space1.eVals(i) = null_eig;
            }

            // Drop the first few leading eigenvectors in the within-class space
            QList<float> eVal_list; eVal_list.reserve(dimsIn);
            float fmax = -1;
            for (int i=0; i<dimsIn; i++) fmax = std::max(fmax, space1.eVals(i));
            for (int i=0; i<dimsIn; i++) eVal_list.append(space1.eVals(i)/fmax);

            QList<float> dSum = Common::CumSum(eVal_list);
            int drop_idx;
            for (drop_idx = 0; drop_idx<dimsIn; drop_idx++)
                if (dSum[drop_idx]/dSum[dimsIn-1] >= directDrop)
                    break;

            drop_idx++;
            space1.keep = dimsIn - drop_idx;

            Eigen::MatrixXf new_vecs = Eigen::MatrixXf(space1.eVecs.rows(), (int)space1.keep);
            Eigen::MatrixXf new_vals = Eigen::MatrixXf((int)space1.keep, 1);

            for (int i = 0; i < space1.keep; i++) {
                new_vecs.col(i) = space1.eVecs.col(i + drop_idx);
                new_vals(i) = space1.eVals(i + drop_idx);
            }

            space1.eVecs = new_vecs;
            space1.eVals = new_vals;

            // We will call this "agressive" whitening. Really, it is not whitening
            // anymore. Instead, we are further scaling the small eigenvalues and the
            // null space eigenvalues (to increase their impact).
            for (int i=0; i<space1.keep; i++) space1.eVecs.col(i) /= pow((double)space1.eVals(i),0.15);
        }

        // Now we project the mean class vectors into this second
        // subspace that minimizes the within-class scatter energy.
        // Inside this subspace we learn a subspace projection that
        // maximizes the between-class scatter energy.
        Eigen::MatrixXd mean2 = Eigen::MatrixXd::Zero(dimsIn, 1);

        // Remove means
        for (int i=0; i<dimsIn; i++)     mean2(i) = classMeans.row(i).sum() / numClasses;
        for (int i=0; i<numClasses; i++) classMeans.col(i) -= mean2;

        // Project into second subspace
        Eigen::MatrixXd data2 = space1.eVecs.transpose().cast<double>() * classMeans;

        // The rank of the between-class scatter matrix is bound by numClasses - 1
        // because each class is a vector used to compute the covariance,
        // but one degree of freedom is lost removing the global mean.
        int dim2 = std::min((int)space1.keep, numClasses-1);
        PCATransform space2;
        space2.keep = dim2;
        space2.trainCore(data2);

        // Compute final projection matrix
        projection = ((space2.eVecs.transpose() * space1.eVecs.transpose()) * pca.eVecs.transpose()).transpose();
        dimsOut = dim2;

        stdDev = 1; // default initialize
        if (isBinary) {
            assert(dimsOut == 1);
            float posVal = 0;
            float negVal = 0;
            Eigen::MatrixXf results(trainingSet.size(),1);
            for (int i = 0; i < trainingSet.size(); i++) {
                Template t;
                project(trainingSet[i],t);
                //Note: the positive class is assumed to be 0 b/c it will
                // typically be the first gallery template in the TemplateList structure
                if (classes[i] == 0)
                    posVal += t.m().at<float>(0,0);
                else if (classes[i] == 1)
                    negVal += t.m().at<float>(0,0);
                else
                    qFatal("Binary mode only supports two class problems.");
                results(i) = t.m().at<float>(0,0);  //used for normalization
            }
            posVal /= classCounts[0];
            negVal /= classCounts[1];

            if (posVal < negVal) {
                //Ensure positive value is supposed to be > 0 after projection
                Eigen::MatrixXf invert = Eigen::MatrixXf::Ones(dimsIn,1);
                invert *= -1;
                projection = invert.transpose() * projection;
            }

            if (normalize)
                stdDev = sqrt(results.array().square().sum() / trainingSet.size());
        }

        qDebug() << "LDA projection dimensions:" << projection.rows() << "->" << projection.cols();
    }

    void project(const Template &src, Template &dst) const
    {
        dst = cv::Mat(1, dimsOut, CV_32FC1);

        // Map Eigen into OpenCV
        Eigen::Map<Eigen::MatrixXf> inMap((float*)src.m().ptr<float>(), src.m().rows*src.m().cols, 1);
        Eigen::Map<Eigen::MatrixXf> outMap(dst.m().ptr<float>(), dimsOut, 1);

        // Do projection
        outMap = projection.transpose() * (inMap - mean);
        if (normalize && isBinary)
            dst.m().at<float>(0,0) = dst.m().at<float>(0,0) / stdDev;
    }

    void store(QDataStream &stream) const
    {
        stream << pcaKeep;
        stream << directLDA;
        stream << directDrop;
        stream << dimsOut;
        stream << mean;
        stream << projection;
        if (normalize && isBinary)
            stream << stdDev;
    }

    void load(QDataStream &stream)
    {
        stream >> pcaKeep;
        stream >> directLDA;
        stream >> directDrop;
        stream >> dimsOut;
        stream >> mean;
        stream >> projection;
        if (normalize && isBinary)
            stream >> stdDev;
    }
};

BR_REGISTER(Transform, LDATransform)

/*!
 * \ingroup transforms
 * \brief Projects input into learned Linear Discriminant Analysis subspace learned on a sparse subset of features with the highest weight in the original LDA algorithm.
 * \author Brendan Klare \cite bklare
 * \br_property float varThreshold BRENDAN FILL ME IN. Default is 1.5.
 * \br_property float pcaKeep BRENDAN FILL ME IN. Default is 0.98.
 * \br_property bool normalize BRENDAN FILL ME IN. Default is true.
 */
class SparseLDATransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(float varThreshold READ get_varThreshold WRITE set_varThreshold RESET reset_varThreshold STORED false)
    Q_PROPERTY(float pcaKeep READ get_pcaKeep WRITE set_pcaKeep RESET reset_pcaKeep STORED false)
    Q_PROPERTY(bool normalize READ get_normalize WRITE set_normalize RESET reset_normalize STORED false)
    BR_PROPERTY(float, varThreshold, 1.5)
    BR_PROPERTY(float, pcaKeep, 0.98)
    BR_PROPERTY(bool, normalize, true)

    LDATransform ldaSparse;
    int dimsOut;
    QList<int> selections;

    Eigen::VectorXf mean;

    void init()
    {
        ldaSparse.init();
        ldaSparse.pcaKeep = pcaKeep;
        ldaSparse.inputVariable = "Label";
        ldaSparse.isBinary = true;
        ldaSparse.normalize = true;
    }

    void train(const TemplateList &_trainingSet)
    {

        LDATransform ldaOrig;
        ldaOrig.init();
        ldaOrig.inputVariable = "Label";
        ldaOrig.pcaKeep = pcaKeep;
        ldaOrig.isBinary = true;
        ldaOrig.normalize = true;

        ldaOrig.train(_trainingSet);

        //Only works on binary class problems for now
        assert(ldaOrig.projection.cols() == 1);
        float ldaStd = EigenUtils::stddev(ldaOrig.projection);
        for (int i = 0; i < ldaOrig.projection.rows(); i++)
            if (abs(int(ldaOrig.projection(i))) > varThreshold * ldaStd)
                selections.append(i);

        TemplateList newSet;
        for (int i = 0; i < _trainingSet.size(); i++) {
            cv::Mat x(_trainingSet[i]);
            cv::Mat y = cv::Mat(selections.size(), 1, CV_32FC1);
            int idx = 0;
            int cnt = 0;
            for (int j = 0; j < x.rows; j++)
                for (int k = 0; k < x.cols; k++, cnt++)
                    if (selections.contains(cnt))
                        y.at<float>(idx++,0) = x.at<float>(j, k);
            newSet.append(Template(_trainingSet[i].file, y));
        }
        ldaSparse.train(newSet);
        dimsOut = ldaSparse.dimsOut;
    }

    void project(const Template &src, Template &dst) const
    {
        Eigen::Map<Eigen::MatrixXf> inMap((float*)src.m().ptr<float>(), src.m().rows*src.m().cols, 1);
        Eigen::Map<Eigen::MatrixXf> outMap(dst.m().ptr<float>(), dimsOut, 1);

        int d = selections.size();
        cv::Mat inSelect(d,1,CV_32F);
        for (int i = 0; i < d; i++)
            inSelect.at<float>(i) = src.m().at<float>(selections[i]);
        ldaSparse.project(Template(src.file, inSelect), dst);
    }

    void store(QDataStream &stream) const
    {
        stream << pcaKeep;
        stream << ldaSparse;
        stream << dimsOut;
        stream << selections;
    }

    void load(QDataStream &stream)
    {
        stream >> pcaKeep;
        stream >> ldaSparse;
        stream >> dimsOut;
        stream >> selections;
    }
};

BR_REGISTER(Transform, SparseLDATransform)

using namespace Eigen;
using namespace cv;

/*!
 * \ingroup transforms
 * \brief Projects input into a within-class minimizing subspace.
 *
 * Like LDA but without the explicit between-class consideration.
 *
 * Note on Compression:
 * Projection matricies can become quite large, resulting in proportionally large model files.
 * WCDA automatically alleviates this issue with lossy compression of the projection matrix.
 * Each element is stored as an 8-bit integer instead of a 32-bit float, resulting in a 75% reduction in the size of the projection matrix.
 * A non-linear (sqrt) scaling is used because element values are distributed around 0, in affect allowing for higher precision storage of the more frequently occuring values.
 *
 * \author Josh Klontz \cite jklontz
 */
class WCDATransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(float keep READ get_keep WRITE set_keep RESET reset_keep STORED false)
    BR_PROPERTY(float, keep, 0.98)

    VectorXf mean;
    MatrixXf projection;

    typedef Matrix<quint8, Dynamic, Dynamic> CompressedMatrix;
    CompressedMatrix compressed;
    float a, b;

    static CompressedMatrix compress(MatrixXf src, float &a, float &b)
    {
        for (int i=0; i<src.rows(); i++)
            for (int j=0; j<src.cols(); j++) {
                float &element = src(i, j);
                element = sqrtf(fabsf(element)) * ((element >= 0) ? 1 : -1);
            }
        b = src.minCoeff();
        a = (src.maxCoeff() - b) / 255;
        return ((src.array() - b) / a).cast<quint8>();
    }

    static MatrixXf decompress(const CompressedMatrix &src, float a, float b)
    {
        MatrixXf dst = src.cast<float>().array() * a + b;
        for (int i=0; i<dst.rows(); i++)
            for (int j=0; j<dst.cols(); j++) {
                float &element = dst(i, j);
                element = (element * element) * ((element >= 0) ? 1 : -1);
            }
        return dst;
    }

    void train(const TemplateList &_templates)
    {
        // Relabel ensures that class values range from 0 to numClasses-1.
        const TemplateList templates = TemplateList::relabel(_templates, "Label", false);
        const int instances = templates.size();
        const int dimsIn = templates.first().m().rows * templates.first().m().cols;

        // Map data into Eigen
        MatrixXf data(dimsIn, instances);
        for (int i=0; i<instances; i++)
            data.col(i) = Map<const VectorXf>(templates[i].m().ptr<float>(), dimsIn);

        // Perform PCA dimensionality reduction
        VectorXf pcaEvals;
        MatrixXf pcaEvecs;
        trainCore(data, keep, mean, pcaEvals, pcaEvecs);
        data = pcaEvecs.transpose() * (data.colwise() - mean);

        // Get ground truth
        const QList<int> classes = File::get<int>(templates, "Label");
        const QMap<int, int> classCounts = templates.countValues<int>("Label");
        const int numClasses = classCounts.size();

        // Compute and remove the class means
        MatrixXf classMeans = MatrixXf::Zero(data.rows(), numClasses);
        for (int i=0; i<instances;  i++) classMeans.col(classes[i]) += data.col(i);
        for (int i=0; i<numClasses; i++) classMeans.col(i) /= classCounts[i];
        for (int i=0; i<instances;  i++) data.col(i) -= classMeans.col(classes[i]);

        // Compute the within-class projection
        VectorXf wcMean, wcEvals;
        MatrixXf wcEvecs;
        trainCore(data, keep, wcMean, wcEvals, wcEvecs);

        // This has the effect of whitening the within-class scatter.
        // In effect, this minimizes the within-class variation energy.
        for (int i=0; i<keep; i++)
            wcEvecs.col(i) /= sqrt(wcEvals(i));

        projection = wcEvecs.transpose() * pcaEvecs.transpose();

        // Compress and restore the projection matrix
        compressed = compress(projection, a, b);
        projection = decompress(compressed, a, b);
    }

    static void trainCore(MatrixXf data, float &keep, VectorXf &mean, VectorXf &eVals, MatrixXf &eVecs)
    {
        const int dimsIn = data.rows();
        const int instances = data.cols();
        const bool dominantEigenEstimation = (dimsIn > instances);

        // Compute and remove mean
        mean = data.rowwise().sum() / instances;
        data.colwise() -= mean;

        // Calculate and normalize covariance matrix
        MatrixXf cov;
        if (dominantEigenEstimation) cov = data.transpose() * data;
        else                         cov = data * data.transpose();
        cov /= (instances-1);

        // Compute eigendecomposition, returning eigenvectors/eigenvalues in increasing order by eigenvalue.
        SelfAdjointEigenSolver<MatrixXf> eSolver(cov);
        VectorXf allEVals = eSolver.eigenvalues();
        MatrixXf allEVecs = eSolver.eigenvectors();
        if (dominantEigenEstimation)
            allEVecs = data * allEVecs;

        if (keep < 1) {
            // Keep eigenvectors that retain a certain energy percentage.
            const float desiredEnergy = keep * allEVals.sum();
            float currentEnergy = 0;
            int i = 0;
            while ((currentEnergy < desiredEnergy) && (i < allEVals.rows())) {
                currentEnergy += allEVals(allEVals.rows()-(i+1));
                i++;
            }
            keep = i;
        } else {
            if (keep > allEVals.rows()) {
                qWarning("Insufficient samples, needed at least %d but only got %d.", (int)keep, (int)allEVals.rows());
                keep = allEVals.rows();
            }
        }

        // Keep highest energy vectors
        eVals = VectorXf((int)keep);
        eVecs = MatrixXf(allEVecs.rows(), (int)keep);
        for (int i=0; i<keep; i++) {
            const int index = allEVals.rows()-(i+1);
            eVals(i) = allEVals(index);
            eVecs.col(i) = allEVecs.col(index) / allEVecs.col(index).norm();
        }
    }

    void project(const Template &src, Template &dst) const
    {
        dst = Mat(1, projection.rows(), CV_32FC1);
        Map<const VectorXf> inMap(src.m().ptr<float>(), src.m().rows * src.m().cols);
        Map<VectorXf> outMap(dst.m().ptr<float>(), projection.rows());
        outMap = projection * (inMap - mean);
    }

    void store(QDataStream &stream) const
    {
        stream << mean << compressed << a << b;
    }

    void load(QDataStream &stream)
    {
        stream >> mean >> compressed >> a >> b;
        projection = decompress(compressed, a, b);
    }
};

BR_REGISTER(Transform, WCDATransform)

} // namespace br

#include "classification/lda.moc"
