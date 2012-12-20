#include <QMap>
#include <QString>
#include <QStringList>
#include <QTime>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <mm_plugin.h>

#include "model.h"
#include "common/opencvutils.h"
#include "common/qtutils.h"
#include "plugins/meta.h"
#include "plugins/regions.h"

//#ifdef MM_SDK_TRAINABLE
#include <boost/smart_ptr.hpp>
#include <exception>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include <error.hpp>
#include <global.hpp>
#include <subset.hpp>
#include <data_intervaller.hpp>
#include <data_splitter.hpp>
#include <data_splitter_5050.hpp>
#include <data_splitter_cv.hpp>
#include <data_splitter_resub.hpp>
#include <data_scaler.hpp>
#include <data_scaler_void.hpp>
#include <data_accessor_splitting_mem.hpp>
#include <criterion_wrapper.hpp>
#include <distance_euclid.hpp>
#include <classifier_knn.hpp>
#include <seq_step_straight_threaded.hpp>
#include <search_seq_dos.hpp>
#include <search_seq_sfs.hpp>
#include <search_seq_sffs.hpp>
#include <search_monte_carlo_threaded.hpp>

using namespace FST;
//#endif // MM_SDK_TRAINABLE

using namespace mm;

enum DimensionStatus {
    On,
    Off,
    Ignore
};

//#ifdef MM_SDK_TRAINABLE
template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
class FST3Data_Accessor_Splitting_MemMM : public Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER>
{
    QList<MatrixList> mll;
    QList<DimensionStatus> dsl;
    int features;
    QMap<int, int> labelCounts;

public:
    typedef Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER> DASM;
    typedef boost::shared_ptr<Data_Scaler<DATATYPE> > PScaler;
    typedef typename DASM::PSplitters PSplitters;

    FST3Data_Accessor_Splitting_MemMM(const QList<MatrixList> &_mll, const QList<DimensionStatus> &_dsl, const PSplitters _dsp, const PScaler _dsc)
        : Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER>("MM", _dsp, _dsc), mll(_mll), dsl(_dsl)
    {
        features = 0;
        foreach (DimensionStatus ds, dsl)
            if (ds != Ignore) features++;
        labelCounts = mll.first().labelCounts();
    }

    FST3Data_Accessor_Splitting_MemMM(const MatrixList &_ml, const PSplitters _dsp, const PScaler _dsc)
        : Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER>("MM", _dsp, _dsc)
    {
        mll.append(_ml);
        features = _ml.first().total() * _ml.first().channels();
        for (int i=0; i<features; i++)
            dsl.append(Off);
        labelCounts = _ml.labelCounts();
    }

    FST3Data_Accessor_Splitting_MemMM* sharing_clone() const;
    virtual std::ostream& print(std::ostream& os) const;

protected:
    FST3Data_Accessor_Splitting_MemMM(const Data_Accessor_Splitting_MemMM &damt, int x)
        : Data_Accessor_Splitting_Mem<DATATYPE,IDXTYPE,INTERVALCONTAINER>(damt, x)
    {} // weak (referencing) copy-constructor to be used in sharing_clone()

    virtual void initial_data_read();    //!< \note off-limits in shared_clone
    virtual void initial_file_prepare() {}

public:
    virtual unsigned int file_getNoOfClasses() const { return labelCounts.size(); }
    virtual unsigned int file_getNoOfFeatures() const { return features; }
    virtual IDXTYPE file_getClassSize(unsigned int cls) const { return labelCounts[cls]; }
};

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
void FST3Data_Accessor_Splitting_MemMM<DATATYPE,IDXTYPE,INTERVALCONTAINER>::initial_data_read() //!< \note off-limits in shared_clone
{
    if (Clonable::is_sharing_clone()) throw fst_error("Data_Accessor_Splitting_MemMM()::initial_data_read() called from shared_clone instance.");
    IDXTYPE idx=0;

    // TODO: Assert that ml data type is DATATYPE
    const QList<float> labels = mll.first().labels();
    foreach (int label, labelCounts.keys()) {
        for (int i=0; i<labels.size(); i++) {
            if (labels[i] == label) {
                int dslIndex = 0;
                foreach (const MatrixList &ml, mll) {
                    const Matrix &m = ml[i];
                    const int dims = m.total() * m.channels();
                    for (int j=0; j<dims; j++)
                        if (dsl[dslIndex++] != Ignore)
                            this->data[idx++] = reinterpret_cast<float*>(m.data)[j];
                }
            }
        }
    }
}

/*template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
Data_Accessor_Splitting_MemMM<DATATYPE,IDXTYPE,INTERVALCONTAINER>* Data_Accessor_Splitting_MemMM<DATATYPE,IDXTYPE,INTERVALCONTAINER>::sharing_clone() const
{
        Data_Accessor_Splitting_MemMM<DATATYPE,IDXTYPE,INTERVALCONTAINER> *clone=new Data_Accessor_Splitting_MemMM<DATATYPE,IDXTYPE,INTERVALCONTAINER>(*this, (int)0);
        clone->set_sharing_cloned();
        return clone;
}

template<typename DATATYPE, typename IDXTYPE, class INTERVALCONTAINER>
std::ostream& Data_Accessor_Splitting_MemMM<DATATYPE,IDXTYPE,INTERVALCONTAINER>::print(std::ostream& os) const
{
        DASM::print(os);
        os << std::endl << "Data_Accessor_Splitting_MemMM()";
        return os;
}*/

//#endif // MM_SDK_TRAINABLE


class FST3DOS : public Feature
{
    friend class Maker<DOS,true>;

    int delta;

    mm::Remap remap;

    DOS(int delta = 1)
    {
        this->delta = delta;
    }

    static QString args()
    {
        return "delta = 1";
    }

    static DOS *make(const QString &args)
    {
        QStringList words = QtUtils::parse(args);
        if (words.size() > 1) qFatal("DOS::make invalid argument count.");

        int delta = 1;

        bool ok;
        switch (words.size()) {
          case 1:
            delta = words[0].toInt(&ok); if (!ok) qFatal("DOS::make expected integer delta.");
        }

        return new DOS(delta);
    }

    QSharedPointer<Feature> clone() const
    {
        return QSharedPointer<Feature>(new DOS(delta));
    }

    void train(const MatrixList &data, Matrix &metadata)
    {
        (void) metadata;
        //#ifdef MM_SDK_TRAINABLE
        try {
            typedef float RETURNTYPE; 	typedef float DATATYPE;       typedef float REALTYPE;
            typedef unsigned int IDXTYPE; typedef unsigned int DIMTYPE; typedef int BINTYPE;
            typedef Subset<BINTYPE, DIMTYPE> SUBSET;
            typedef Data_Intervaller<std::vector<Data_Interval<IDXTYPE> >,IDXTYPE> INTERVALLER;
            typedef boost::shared_ptr<Data_Splitter<INTERVALLER,IDXTYPE> > PSPLITTER;
            typedef Data_Splitter_CV<INTERVALLER,IDXTYPE> SPLITTERCV;
            typedef Data_Splitter_5050<INTERVALLER,IDXTYPE> SPLITTER5050;
            typedef Data_Splitter_Resub<INTERVALLER,IDXTYPE> SPLITTERRESUB;
            typedef Data_Accessor_Splitting_MemMM<DATATYPE,IDXTYPE,INTERVALLER> DATAACCESSOR;
            typedef Distance_Euclid<DATATYPE,DIMTYPE,SUBSET> DISTANCE;
            typedef Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE> CLASSIFIERKNN;
            typedef Criterion_Wrapper<RETURNTYPE,SUBSET,CLASSIFIERKNN,DATAACCESSOR> WRAPPERKNN;
            typedef Sequential_Step_Straight_Threaded<RETURNTYPE,DIMTYPE,SUBSET,WRAPPERKNN,24> EVALUATOR;

            // Initialize dataset
            PSPLITTER dsp_outer(new SPLITTER5050()); // keep second half of data for independent testing of final classification performance
            PSPLITTER dsp_inner(new SPLITTERCV(3)); // in the course of search use the first half of data by 3-fold cross-validation in wrapper FS criterion evaluation
            boost::shared_ptr<Data_Scaler<DATATYPE> > dsc(new Data_Scaler_void<DATATYPE>()); // do not scale data
            boost::shared_ptr<std::vector<PSPLITTER> > splitters(new std::vector<PSPLITTER>); // set-up data access
            splitters->push_back(dsp_outer); //splitters->push_back(dsp_inner);
            boost::shared_ptr<DATAACCESSOR> da(new DATAACCESSOR(data, splitters, dsc));
            da->initialize();
            da->setSplittingDepth(0); if(!da->getFirstSplit()) throw fst_error("50/50 data split failed.");
            //da->setSplittingDepth(1); if(!da->getFirstSplit()) throw fst_error("3-fold cross-validation failure.");
            boost::shared_ptr<SUBSET> sub(new SUBSET(da->getNoOfFeatures())); // initiate the storage for subset to-be-selected
            //sub->select_all();

            // Run search
            boost::shared_ptr<CLASSIFIERKNN> cknn(new CLASSIFIERKNN); cknn->set_k(1);
            boost::shared_ptr<WRAPPERKNN> wknn(new WRAPPERKNN);
            wknn->initialize(cknn,da);
            boost::shared_ptr<EVALUATOR> eval(new EVALUATOR); // set-up the standard sequential search step object (option: hybrid, ensemble, etc.)
            //Search_DOS<RETURNTYPE,DIMTYPE,SUBSET,WRAPPERKNN,EVALUATOR> srch(eval); // set-up Sequential Forward Floating Selection search procedure
            //srch.set_delta(delta);

            //FST::Search_SFFS<RETURNTYPE,DIMTYPE,SUBSET,WRAPPERKNN,EVALUATOR> srch(eval);
            //srch.set_search_direction(FST::BACKWARD);

            //FST::Search_SFS<RETURNTYPE,DIMTYPE,SUBSET,WRAPPERKNN,EVALUATOR> srch(eval);
            //srch.set_search_direction(FST::FORWARD);

            FST::Search_Monte_Carlo_Threaded<RETURNTYPE,DIMTYPE,SUBSET,WRAPPERKNN,24> srch;
            srch.set_cardinality_randomization(0.5); // probability of inclusion of each particular feature (~implies also the expected subset size)
            srch.set_stopping_condition(0/*max trials*/,30/*seconds*/); // one or both values must have positive value

            RETURNTYPE critval_train;
            if(!srch.search(0,critval_train,sub,wknn,std::cout)) throw fst_error("Search not finished.");

            // Create map matrix
            const int dims = sub->get_d_raw();
            cv::Mat xMap(1, dims, CV_16SC1),
                    yMap(1, dims, CV_16SC1);
            int index = 0;
            for (int i=0; i<dims; i++) {
                if (sub->selected_raw(i)) {
                    xMap.at<short>(0, index) = i;
                    yMap.at<short>(0, index) = 0;
                    index++;
                }
            }

            remap = Remap(xMap, yMap, cv::INTER_NEAREST);
        }
        catch (fst_error &e) { qFatal("FST ERROR: %s, code=%d", e.what(), e.code()); }
        catch (std::exception &e) { qFatal("non-FST ERROR: %s", e.what()); }
        metadata >> remap;
        //#else // MM_SDK_TRAINABLE
        //qFatal("StreamwiseFS::train not supported.");
        //#endif // MM_SDK_TRAINABLE
    }

    void project(const Matrix &src, Matrix &dst) const
    {
        dst = src;
        dst >> remap;
    }

    void store(QDataStream &stream) const
    {
        stream << remap;
    }

    void load(QDataStream &stream)
    {
        stream >> remap;
    }
};

MM_REGISTER(Feature, FST3DOS, true)


class FST3StreamwiseFS : public Feature
{
    friend class Maker<StreamwiseFS,true>;

    QSharedPointer<Feature> weakLearnerTemplate;
    int time;

    mm::Dup dup;
    mm::Remap remap;

    StreamwiseFS(const QSharedPointer<Feature> &weakLearnerTemplate, int time)
        : dup(weakLearnerTemplate, 1)
    {
        this->weakLearnerTemplate = weakLearnerTemplate;
        this->time = time;
    }

    static QString args()
    {
        return "<feature> weakLearnerTemplate, int time";
    }

    static StreamwiseFS *make(const QString &args)
    {
        QStringList words = QtUtils::parse(args);
        if (words.size() != 2) qFatal("StreamwiseFS::make invalid argument count.");

        QSharedPointer<Feature> weakLearnerTemplate = Feature::make(words[0]);
        bool ok;
        int time = words[1].toInt(&ok); assert(ok);

        return new StreamwiseFS(weakLearnerTemplate, time);
    }

    QSharedPointer<Feature> clone() const
    {
        return QSharedPointer<Feature>(new StreamwiseFS(weakLearnerTemplate, time));
    }

    void train(const MatrixList &data, Matrix &metadata)
    {
        QList< QSharedPointer<Feature> > weakLearners;
        QList<MatrixList> projectedDataList;
        QList<int> weakLearnerDimsList;
        QList<DimensionStatus> dimStatusList;

        QTime timer; timer.start();
        while (timer.elapsed() / 1000 < time) {
            // Construct a new weak learner
            QSharedPointer<Feature> newWeakLearner = weakLearnerTemplate->clone();
            Matrix metadataCopy(metadata);
            newWeakLearner->train(data, metadataCopy);
            weakLearners.append(newWeakLearner);

            MatrixList projectedData = data;
            projectedData >> *newWeakLearner;
            projectedDataList.append(projectedData);
            weakLearnerDimsList.append(projectedData.first().total() * projectedData.first().channels());
            for (int i=0; i<weakLearnerDimsList.last(); i++) dimStatusList.append(Off);

            //#ifdef MM_SDK_TRAINABLE
            try
            {
                typedef float RETURNTYPE; 	typedef float DATATYPE;       typedef float REALTYPE;
                typedef unsigned int IDXTYPE; typedef unsigned int DIMTYPE; typedef int BINTYPE;
                typedef Subset<BINTYPE, DIMTYPE> SUBSET;
                typedef Data_Intervaller<std::vector<Data_Interval<IDXTYPE> >,IDXTYPE> INTERVALLER;
                typedef boost::shared_ptr<Data_Splitter<INTERVALLER,IDXTYPE> > PSPLITTER;
                typedef Data_Splitter_CV<INTERVALLER,IDXTYPE> SPLITTERCV;
                typedef Data_Splitter_5050<INTERVALLER,IDXTYPE> SPLITTER5050;
                typedef Data_Accessor_Splitting_MemMM<DATATYPE,IDXTYPE,INTERVALLER> DATAACCESSOR;
                typedef Distance_Euclid<DATATYPE,DIMTYPE,SUBSET> DISTANCE;
                typedef Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE> CLASSIFIERKNN;
                typedef Criterion_Wrapper<RETURNTYPE,SUBSET,CLASSIFIERKNN,DATAACCESSOR> WRAPPERKNN;
                typedef Sequential_Step_Straight_Threaded<RETURNTYPE,DIMTYPE,SUBSET,WRAPPERKNN,24> EVALUATOR;

                // Initialize dataset
                PSPLITTER dsp_outer(new SPLITTER5050()); // keep second half of data for independent testing of final classification performance
                PSPLITTER dsp_inner(new SPLITTERCV(3)); // in the course of search use the first half of data by 3-fold cross-validation in wrapper FS criterion evaluation
                boost::shared_ptr<Data_Scaler<DATATYPE> > dsc(new Data_Scaler_void<DATATYPE>()); // do not scale data
                boost::shared_ptr<std::vector<PSPLITTER> > splitters(new std::vector<PSPLITTER>); // set-up data access
                splitters->push_back(dsp_outer); splitters->push_back(dsp_inner);
                boost::shared_ptr<DATAACCESSOR> da(new DATAACCESSOR(projectedDataList, dimStatusList, splitters, dsc));
                da->initialize();
                da->setSplittingDepth(0); if(!da->getFirstSplit()) throw fst_error("50/50 data split failed.");
                da->setSplittingDepth(1); if(!da->getFirstSplit()) throw fst_error("3-fold cross-validation failure.");
                boost::shared_ptr<SUBSET> sub(new SUBSET(da->getNoOfFeatures())); // initiate the storage for subset to-be-selected

                { // Initialize subset from previous iteration results
                    sub->deselect_all();
                    int index = 0;
                    for (int i=0; i<dimStatusList.size(); i++) {
                        if (dimStatusList[i] == On) sub->select(index);
                        if (dimStatusList[i] != Ignore) index++;
                    }
                }

                // Run search
                boost::shared_ptr<CLASSIFIERKNN> cknn(new CLASSIFIERKNN); cknn->set_k(3); // set-up 3-Nearest Neighbor classifier based on Euclidean distances
                boost::shared_ptr<WRAPPERKNN> wknn(new WRAPPERKNN); // wrap the 3-NN classifier to enable its usage as FS criterion (criterion value will be estimated by 3-fold cross-val.)
                wknn->initialize(cknn,da);
                boost::shared_ptr<EVALUATOR> eval(new EVALUATOR); // set-up the standard sequential search step object (option: hybrid, ensemble, etc.)
                Search_DOS<RETURNTYPE,DIMTYPE,SUBSET,WRAPPERKNN,EVALUATOR> srch(eval); // set-up Sequential Forward Floating Selection search procedure
                srch.set_delta(1);
                RETURNTYPE critval_train;
                if(!srch.search(0,critval_train,sub,wknn,std::cout)) throw fst_error("Search not finished.");

                { // Update results
                    int dslIndex = dimStatusList.size() - 1;
                    int subIndex = da->getNoOfFeatures() - 1;
                    for (int wlIndex = weakLearnerDimsList.size()-1; wlIndex >= 0; wlIndex--) {
                        const int weakLearnerDims = weakLearnerDimsList[wlIndex];
                        int numSelectedDims = 0;
                        for (int i=0; i<weakLearnerDims; i++) {
                            if (dimStatusList[dslIndex] != Ignore)
                                dimStatusList[dslIndex] = sub->selected_raw(subIndex--) ? numSelectedDims++, On : Ignore;
                            dslIndex--;
                        }

                        if (numSelectedDims == 0) {
                            for (int j=0; j<weakLearnerDims; j++)
                                dimStatusList.removeAt(dslIndex+1);
                            weakLearnerDimsList.removeAt(wlIndex);
                            projectedDataList.removeAt(wlIndex);
                            weakLearners.removeAt(wlIndex);
                        }
                    }
                }
            }
            catch (fst_error &e) { qFatal("FST ERROR: %s, code=%d", e.what(), e.code()); }
            catch (std::exception &e) { qFatal("non-FST ERROR: %s", e.what()); }
            //#else // MM_SDK_TRAINABLE
            //qFatal("StreamwiseFS::train not supported.");
            //#endif // MM_SDK_TRAINABLE
        }

        dup = Dup(weakLearners);

        // Create map matrix
        int dims = 0;
        foreach (DimensionStatus ds, dimStatusList) if (ds == On) dims++;
        cv::Mat xMap(1, dims, CV_16SC1),
                yMap(1, dims, CV_16SC1);
        int index = 0;
        for (int i=0; i<dimStatusList.size(); i++) {
            if (dimStatusList[i] == On) {
                xMap.at<short>(0, index) = i;
                yMap.at<short>(0, index) = 0;
                index++;
            }
        }

        remap = Remap(xMap, yMap, cv::INTER_NEAREST);
    }

    void project(const Matrix &src, Matrix &dst) const
    {
        dst = src;
        dst >> dup >> mm::Cat >> remap;
    }

    void store(QDataStream &stream) const
    {
        stream << dup << remap;
    }

    void load(QDataStream &stream)
    {
        stream >> dup >> remap;
    }
};

MM_REGISTER(Feature, FST3StreamwiseFS, true)
