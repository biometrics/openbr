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
#include "openbr_internal.h"

#include "openbr/core/common.h"
#include "openbr/core/eigenutils.h"

namespace br
{

/*!
 * \ingroup transforms
 * \brief Projects input into learned Principal Component Analysis subspace.
 * \author Brendan Klare \cite bklare
 * \author Josh Klontz \cite jklontz
 */
class PCATransform : public Transform
{
    Q_OBJECT
    friend class DFFSTransform;
    friend class LDATransform;

protected:
    Q_PROPERTY(float keep READ get_keep WRITE set_keep RESET reset_keep STORED false)
    Q_PROPERTY(int drop READ get_drop WRITE set_drop RESET reset_drop STORED false)
    Q_PROPERTY(bool whiten READ get_whiten WRITE set_whiten RESET reset_whiten STORED false)

    /*!
     *     keep <  0: All eigenvalues are retained.
     *     keep =  0: No PCA performed, eigenvectors form an identity matrix.
     * 0 < keep <  1: Fraction of the variance to retain.
     *     keep >= 1: Number of leading eigenvectors to retain.
     */
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
            if (keep + drop > allEVals.rows())
                qFatal("Insufficient samples, needed at least %d but only got %d.", (int)keep + drop, (int)allEVals.rows());
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
 * \brief PCA on each row.
 * \author Josh Klontz \cite jklontz
 */
class RowWisePCATransform : public PCATransform
{
    Q_OBJECT

    void train(const TemplateList &trainingSet)
    {
        if (trainingSet.first().m().type() != CV_32FC1)
            qFatal("Requires single channel 32-bit floating point matrices.");

        originalRows = trainingSet.first().m().rows;
        const int dimsIn = trainingSet.first().m().cols;
        int instances = 0;
        foreach (const Template &t, trainingSet)
            instances += t.m().rows;

        // Map into 64-bit Eigen matrix
        Eigen::MatrixXd data(dimsIn, instances);
        int index = 0;
        foreach (const Template &t, trainingSet)
            for (int i=0; i<t.m().rows; i++)
                data.col(index++) = Eigen::Map<const Eigen::MatrixXf>(t.m().ptr<float>(i), dimsIn, 1).cast<double>();

        PCATransform::trainCore(data);
    }

    void project(const Template &src, Template &dst) const
    {
        dst = cv::Mat(src.m().rows, keep, CV_32FC1);

        for (int i=0; i<src.m().rows; i++) {
            Eigen::Map<const Eigen::MatrixXf> inMap(src.m().ptr<float>(i), src.m().cols, 1);
            Eigen::Map<Eigen::MatrixXf> outMap(dst.m().ptr<float>(i), keep, 1);
            outMap = eVecs.transpose() * (inMap - mean);
        }
    }
};

BR_REGISTER(Transform, RowWisePCATransform)

/*!
 * \ingroup transforms
 * \brief Computes Distance From Feature Space (DFFS) \cite moghaddam97.
 * \author Josh Klontz \cite jklontz
 */
class DFFSTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(float keep READ get_keep WRITE set_keep RESET reset_keep STORED false)
    BR_PROPERTY(float, keep, 0.95)

    PCATransform pca;
    Transform *cvtFloat;

    void init()
    {
        pca.keep = keep;
        cvtFloat = make("CvtFloat");
    }

    void train(const TemplateList &data)
    {
        pca.train((*cvtFloat)(data));
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        dst.file.set("DFFS", sqrt(pca.residualReconstructionError((*cvtFloat)(src))));
    }

    void store(QDataStream &stream) const
    {
        pca.store(stream);
    }

    void load(QDataStream &stream)
    {
        pca.load(stream);
    }
};

BR_REGISTER(Transform, DFFSTransform)

/*!
 * \ingroup transforms
 * \brief Projects input into learned Linear Discriminant Analysis subspace.
 * \author Brendan Klare \cite bklare
 * \author Josh Klontz \cite jklontz
 */
class LDATransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(float pcaKeep READ get_pcaKeep WRITE set_pcaKeep RESET reset_pcaKeep STORED false)
    Q_PROPERTY(bool pcaWhiten READ get_pcaWhiten WRITE set_pcaWhiten RESET reset_pcaWhiten STORED false)
    Q_PROPERTY(int directLDA READ get_directLDA WRITE set_directLDA RESET reset_directLDA STORED false)
    Q_PROPERTY(float directDrop READ get_directDrop WRITE set_directDrop RESET reset_directDrop STORED false)
    Q_PROPERTY(QString inputVariable READ get_inputVariable WRITE set_inputVariable RESET reset_inputVariable STORED false)
    Q_PROPERTY(bool isBinary READ get_isBinary WRITE set_isBinary RESET reset_isBinary STORED true)
    Q_PROPERTY(bool normalize READ get_normalize WRITE set_normalize RESET reset_normalize STORED true)
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
    bool trained;

    void init()
    {
        trained = false;
    }

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

        if (isBinary) {
            assert(dimsOut == 1);
            TemplateList projected;
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

        trained = true;
    }

    void project(const Template &src, Template &dst) const
    {
        dst = cv::Mat(1, dimsOut, CV_32FC1);

        // Map Eigen into OpenCV
        Eigen::Map<Eigen::MatrixXf> inMap((float*)src.m().ptr<float>(), src.m().rows*src.m().cols, 1);
        Eigen::Map<Eigen::MatrixXf> outMap(dst.m().ptr<float>(), dimsOut, 1);

        // Do projection
        outMap = projection.transpose() * (inMap - mean);

        if (normalize && isBinary && trained)
            dst.m().at<float>(0,0) = dst.m().at<float>(0,0) / stdDev;
    }

    void store(QDataStream &stream) const
    {
        stream << pcaKeep << directLDA << directDrop << dimsOut << mean << projection << stdDev << normalize << isBinary << trained;
    }

    void load(QDataStream &stream)
    {
        stream >> pcaKeep >> directLDA >> directDrop >> dimsOut >> mean >> projection >> stdDev >> normalize >> isBinary >> trained;
    }
};

BR_REGISTER(Transform, LDATransform)

/*!
 * \ingroup distances
 * \brief L1 distance computed using eigen.
 * \author Josh Klontz \cite jklontz
 */
class L1Distance : public Distance
{
    Q_OBJECT

    float compare(const Template &a, const Template &b) const
    {
        const int size = a.m().rows * a.m().cols;
        Eigen::Map<Eigen::VectorXf> aMap((float*)a.m().data, size);
        Eigen::Map<Eigen::VectorXf> bMap((float*)b.m().data, size);
        return (aMap-bMap).cwiseAbs().sum();
    }
};

BR_REGISTER(Distance, L1Distance)

/*!
 * \ingroup distances
 * \brief L2 distance computed using eigen.
 * \author Josh Klontz \cite jklontz
 */
class L2Distance : public Distance
{
    Q_OBJECT

    float compare(const Template &a, const Template &b) const
    {
        const int size = a.m().rows * a.m().cols;
        Eigen::Map<Eigen::VectorXf> aMap((float*)a.m().data, size);
        Eigen::Map<Eigen::VectorXf> bMap((float*)b.m().data, size);
        return (aMap-bMap).squaredNorm();
    }
};

BR_REGISTER(Distance, L2Distance)

} // namespace br

#include "eigen3.moc"
