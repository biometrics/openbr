#include "cascade.h"
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace br;
using namespace cv;

bool CascadeImageReader::create( const string _posFilename, const string _negFilename, Size _winSize )
{
    return posReader.create(_posFilename) && negReader.create(_negFilename, _winSize);
}

CascadeImageReader::NegReader::NegReader()
{
    src.create( 0, 0 , CV_8UC1 );
    img.create( 0, 0, CV_8UC1 );
    point = offset = Point( 0, 0 );
    scale       = 1.0F;
    scaleFactor = 1.4142135623730950488016887242097F;
    stepFactor  = 0.5F;
}

bool CascadeImageReader::NegReader::create( const string _filename, Size _winSize )
{
    string dirname, str;
    std::ifstream file(_filename.c_str());
    if ( !file.is_open() )
        return false;

    size_t pos = _filename.rfind('\\');
    char dlmrt = '\\';
    if (pos == string::npos)
    {
        pos = _filename.rfind('/');
        dlmrt = '/';
    }
    dirname = pos == string::npos ? "" : _filename.substr(0, pos) + dlmrt;
    while( !file.eof() )
    {
        std::getline(file, str);
        if (str.empty()) break;
        if (str.at(0) == '#' ) continue; /* comment */
        imgFilenames.push_back(dirname + str);
    }
    file.close();

    winSize = _winSize;
    last = round = 0;
    return true;
}

bool CascadeImageReader::NegReader::nextImg()
{
    Point _offset = Point(0,0);
    size_t count = imgFilenames.size();
    for( size_t i = 0; i < count; i++ )
    {
        src = imread( imgFilenames[last++], 0 );
        if( src.empty() )
            continue;
        round += last / count;
        round = round % (winSize.width * winSize.height);
        last %= count;

        _offset.x = std::min( (int)round % winSize.width, src.cols - winSize.width );
        _offset.y = std::min( (int)round / winSize.width, src.rows - winSize.height );
        if( !src.empty() && src.type() == CV_8UC1
                && _offset.x >= 0 && _offset.y >= 0 )
            break;
    }

    if( src.empty() )
        return false; // no appropriate image
    point = offset = _offset;
    scale = max( ((float)winSize.width + point.x) / ((float)src.cols),
                 ((float)winSize.height + point.y) / ((float)src.rows) );

    Size sz( (int)(scale*src.cols + 0.5F), (int)(scale*src.rows + 0.5F) );
    resize( src, img, sz );
    return true;
}

bool CascadeImageReader::NegReader::get( Mat& _img )
{
    CV_Assert( !_img.empty() );
    CV_Assert( _img.type() == CV_8UC1 );
    CV_Assert( _img.cols == winSize.width );
    CV_Assert( _img.rows == winSize.height );

    if( img.empty() )
        if ( !nextImg() )
            return false;

    Mat mat( winSize.height, winSize.width, CV_8UC1,
        (void*)(img.data + point.y * img.step + point.x * img.elemSize()), img.step );
    mat.copyTo(_img);

    if( (int)( point.x + (1.0F + stepFactor ) * winSize.width ) < img.cols )
        point.x += (int)(stepFactor * winSize.width);
    else
    {
        point.x = offset.x;
        if( (int)( point.y + (1.0F + stepFactor ) * winSize.height ) < img.rows )
            point.y += (int)(stepFactor * winSize.height);
        else
        {
            point.y = offset.y;
            scale *= scaleFactor;
            if( scale <= 1.0F )
                resize( src, img, Size( (int)(scale*src.cols), (int)(scale*src.rows) ) );
            else
            {
                if ( !nextImg() )
                    return false;
            }
        }
    }
    return true;
}

CascadeImageReader::PosReader::PosReader()
{
    file = 0;
    vec = 0;
}

bool CascadeImageReader::PosReader::create( const string _filename )
{
    if ( file )
        fclose( file );
    file = fopen( _filename.c_str(), "rb" );

    if( !file )
        return false;
    short tmp = 0;
    if( fread( &count, sizeof( count ), 1, file ) != 1 ||
        fread( &vecSize, sizeof( vecSize ), 1, file ) != 1 ||
        fread( &tmp, sizeof( tmp ), 1, file ) != 1 ||
        fread( &tmp, sizeof( tmp ), 1, file ) != 1 )
        CV_Error_( CV_StsParseError, ("wrong file format for %s\n", _filename.c_str()) );
    base = sizeof( count ) + sizeof( vecSize ) + 2*sizeof( tmp );
    if( feof( file ) )
        return false;
    last = 0;
    vec = (short*) cvAlloc( sizeof( *vec ) * vecSize );
    CV_Assert( vec );
    return true;
}

bool CascadeImageReader::PosReader::get( Mat &_img )
{
    CV_Assert( _img.rows * _img.cols == vecSize );
    uchar tmp = 0;
    size_t elements_read = fread( &tmp, sizeof( tmp ), 1, file );
    if( elements_read != 1 )
        CV_Error( CV_StsBadArg, "Can not get new positive sample. The most possible reason is "
                                "insufficient count of samples in given vec-file.\n");
    elements_read = fread( vec, sizeof( vec[0] ), vecSize, file );
    if( elements_read != (size_t)(vecSize) )
        CV_Error( CV_StsBadArg, "Can not get new positive sample. Seems that vec-file has incorrect structure.\n");

    if( feof( file ) || last++ >= count )
        CV_Error( CV_StsBadArg, "Can not get new positive sample. vec-file is over.\n");

    for( int r = 0; r < _img.rows; r++ )
    {
        for( int c = 0; c < _img.cols; c++ )
            _img.ptr(r)[c] = (uchar)vec[r * _img.cols + c];
    }
    return true;
}

void CascadeImageReader::PosReader::restart()
{
    CV_Assert( file );
    last = 0;
    fseek( file, base, SEEK_SET );
}

CascadeImageReader::PosReader::~PosReader()
{
    if (file)
        fclose( file );
    cvFree( &vec );
}

// -------------------------------------- Cascade --------------------------------------------

static const char* stageTypes[] = { CC_BOOST };
static const char* featureTypes[] = { CC_LBP, CC_HAAR, CC_HOG, CC_HOGMULTI, CC_NPD };

CascadeParams::CascadeParams() : stageType( defaultStageType ),
    featureType( defaultFeatureType ), winSize( cvSize(24, 24) )
{
    name = CC_CASCADE_PARAMS;
}
CascadeParams::CascadeParams( int _stageType, int _featureType ) : stageType( _stageType ),
    featureType( _featureType ), winSize( cvSize(24, 24) )
{
    name = CC_CASCADE_PARAMS;
}

//---------------------------- CascadeParams --------------------------------------

void CascadeParams::write( FileStorage &fs ) const
{
    string stageTypeStr = stageType == BOOST ? CC_BOOST : string();
    CV_Assert( !stageTypeStr.empty() );
    fs << CC_STAGE_TYPE << stageTypeStr;
    string featureTypeStr = featureType == FeatureParams::LBP ? CC_HAAR :
                            0;
    CV_Assert( !stageTypeStr.empty() );
    fs << CC_FEATURE_TYPE << featureTypeStr;
    fs << CC_HEIGHT << winSize.height;
    fs << CC_WIDTH << winSize.width;
}

bool CascadeParams::read( const FileNode &node )
{
    if ( node.empty() )
        return false;
    string stageTypeStr, featureTypeStr;
    FileNode rnode = node[CC_STAGE_TYPE];
    if ( !rnode.isString() )
        return false;
    rnode >> stageTypeStr;
    stageType = !stageTypeStr.compare( CC_BOOST ) ? BOOST : -1;
    if (stageType == -1)
        return false;
    rnode = node[CC_FEATURE_TYPE];
    if ( !rnode.isString() )
        return false;
    rnode >> featureTypeStr;
    featureType = !featureTypeStr.compare( CC_LBP ) ? FeatureParams::LBP :
                  -1;
    if (featureType == -1)
        return false;
    node[CC_HEIGHT] >> winSize.height;
    node[CC_WIDTH] >> winSize.width;
    return winSize.height > 0 && winSize.width > 0;
}

void CascadeParams::printDefaults() const
{
    Params::printDefaults();
    cout << "  [-stageType <";
    for( int i = 0; i < (int)(sizeof(stageTypes)/sizeof(stageTypes[0])); i++ )
    {
        cout << (i ? " | " : "") << stageTypes[i];
        if ( i == defaultStageType )
            cout << "(default)";
    }
    cout << ">]" << endl;

    cout << "  [-featureType <{";
    for( int i = 0; i < (int)(sizeof(featureTypes)/sizeof(featureTypes[0])); i++ )
    {
        cout << (i ? ", " : "") << featureTypes[i];
        if ( i == defaultStageType )
            cout << "(default)";
    }
    cout << "}>]" << endl;
    cout << "  [-w <sampleWidth = " << winSize.width << ">]" << endl;
    cout << "  [-h <sampleHeight = " << winSize.height << ">]" << endl;
}

void CascadeParams::printAttrs() const
{
    cout << "stageType: " << stageTypes[stageType] << endl;
    cout << "featureType: " << featureTypes[featureType] << endl;
    cout << "sampleWidth: " << winSize.width << endl;
    cout << "sampleHeight: " << winSize.height << endl;
}

bool CascadeParams::scanAttr( const string prmName, const string val )
{
    bool res = true;
    if( !prmName.compare( "-stageType" ) )
    {
        for( int i = 0; i < (int)(sizeof(stageTypes)/sizeof(stageTypes[0])); i++ )
            if( !val.compare( stageTypes[i] ) )
                stageType = i;
    }
    else if( !prmName.compare( "-featureType" ) )
    {
        for( int i = 0; i < (int)(sizeof(featureTypes)/sizeof(featureTypes[0])); i++ )
            if( !val.compare( featureTypes[i] ) )
                featureType = i;
    }
    else if( !prmName.compare( "-w" ) )
    {
        winSize.width = atoi( val.c_str() );
    }
    else if( !prmName.compare( "-h" ) )
    {
        winSize.height = atoi( val.c_str() );
    }
    else
        res = false;
    return res;
}

//---------------------------- CascadeClassifier --------------------------------------

bool BrCascadeClassifier::train( const string _cascadeDirName,
                                const string _posFilename,
                                const string _negFilename,
                                int _numPos, int _numNeg,
                                int _precalcValBufSize, int _precalcIdxBufSize,
                                int _numStages,
                                const CascadeParams& _cascadeParams,
                                const FeatureParams& _featureParams,
                                const CascadeBoostParams& _stageParams,
                                bool baseFormatSave )
{
    // Start recording clock ticks for training time output
    const clock_t begin_time = clock();

    if( _cascadeDirName.empty() || _posFilename.empty() || _negFilename.empty() )
        CV_Error( CV_StsBadArg, "_cascadeDirName or _bgfileName or _vecFileName is NULL" );

    string dirName;
    if (_cascadeDirName.find_last_of("/\\") == (_cascadeDirName.length() - 1) )
        dirName = _cascadeDirName;
    else
        dirName = _cascadeDirName + '/';

    numPos = _numPos;
    numNeg = _numNeg;
    numStages = _numStages;
    if ( !imgReader.create( _posFilename, _negFilename, _cascadeParams.winSize ) )
    {
        cout << "Image reader can not be created from -vec " << _posFilename
                << " and -bg " << _negFilename << "." << endl;
        return false;
    }
    if ( !load( dirName ) )
    {
        cascadeParams = _cascadeParams;
        featureParams = FeatureParams::create(cascadeParams.featureType);
        featureParams->init(_featureParams);
        stageParams = new CascadeBoostParams;
        *stageParams = _stageParams;
        featureEvaluator = FeatureEvaluator::create(cascadeParams.featureType);
        featureEvaluator->init( (FeatureParams*)featureParams, numPos + numNeg, cascadeParams.winSize );
        stageClassifiers.reserve( numStages );
    }
    cout << "PARAMETERS:" << endl;
    cout << "cascadeDirName: " << _cascadeDirName << endl;
    cout << "vecFileName: " << _posFilename << endl;
    cout << "bgFileName: " << _negFilename << endl;
    cout << "numPos: " << _numPos << endl;
    cout << "numNeg: " << _numNeg << endl;
    cout << "numStages: " << numStages << endl;
    cout << "precalcValBufSize[Mb] : " << _precalcValBufSize << endl;
    cout << "precalcIdxBufSize[Mb] : " << _precalcIdxBufSize << endl;
    cascadeParams.printAttrs();
    stageParams->printAttrs();
    featureParams->printAttrs();

    int startNumStages = (int)stageClassifiers.size();
    if ( startNumStages > 1 )
        cout << endl << "Stages 0-" << startNumStages-1 << " are loaded" << endl;
    else if ( startNumStages == 1)
        cout << endl << "Stage 0 is loaded" << endl;

    double requiredLeafFARate = pow( (double) stageParams->maxFalseAlarm, (double) numStages ) /
                                (double)stageParams->max_depth;
    double tempLeafFARate;

    for( int i = startNumStages; i < numStages; i++ )
    {
        cout << endl << "===== TRAINING " << i << "-stage =====" << endl;
        cout << "<BEGIN" << endl;
        if ( !updateTrainingSet( tempLeafFARate ) )
        {
            cout << "Train dataset for temp stage can not be filled. "
                "Branch training terminated." << endl;
            break;
        }
        if( tempLeafFARate <= requiredLeafFARate )
        {
            cout << "Required leaf false alarm rate achieved. "
                 "Branch training terminated." << endl;
            break;
        }

        CascadeBoost* tempStage = new CascadeBoost;
        bool isStageTrained = tempStage->train( (FeatureEvaluator*)featureEvaluator,
                                                curNumSamples, _precalcValBufSize, _precalcIdxBufSize,
                                                *((CascadeBoostParams*)stageParams) );
        cout << "END>" << endl;

        if(!isStageTrained)
            break;

        stageClassifiers.push_back( tempStage );

        // save params
        if( i == 0)
        {
            std::string paramsFilename = dirName + CC_PARAMS_FILENAME;
            FileStorage fs( paramsFilename, FileStorage::WRITE);
            if ( !fs.isOpened() )
            {
                cout << "Parameters can not be written, because file " << paramsFilename
                        << " can not be opened." << endl;
                return false;
            }
            fs << FileStorage::getDefaultObjectName(paramsFilename) << "{";
            writeParams( fs );
            fs << "}";
        }
        // save current stage
        char buf[10];
        sprintf(buf, "%s%d", "stage", i );
        string stageFilename = dirName + buf + ".xml";
        FileStorage fs( stageFilename, FileStorage::WRITE );
        if ( !fs.isOpened() )
        {
            cout << "Current stage can not be written, because file " << stageFilename
                    << " can not be opened." << endl;
            return false;
        }
        fs << FileStorage::getDefaultObjectName(stageFilename) << "{";
        tempStage->write( fs, Mat() );
        fs << "}";

        // Output training time up till now
        float seconds = float( clock () - begin_time ) / CLOCKS_PER_SEC;
        int days = int(seconds) / 60 / 60 / 24;
        int hours = (int(seconds) / 60 / 60) % 24;
        int minutes = (int(seconds) / 60) % 60;
        int seconds_left = int(seconds) % 60;
        cout << "Training until now has taken " << days << " days " << hours << " hours " << minutes << " minutes " << seconds_left <<" seconds." << endl;
    }

    if(stageClassifiers.size() == 0)
    {
        cout << "Cascade classifier can't be trained. Check the used training parameters." << endl;
        return false;
    }

    save( dirName + CC_CASCADE_FILENAME, baseFormatSave );

    return true;
}

int BrCascadeClassifier::predict( int sampleIdx )
{
    CV_DbgAssert( sampleIdx < numPos + numNeg );
    for (vector< Ptr<CascadeBoost> >::iterator it = stageClassifiers.begin();
        it != stageClassifiers.end(); it++ )
    {
        if ( (*it)->predict( sampleIdx ) == 0.f )
            return 0;
    }
    return 1;
}

bool BrCascadeClassifier::updateTrainingSet( double& acceptanceRatio)
{
    int64 posConsumed = 0, negConsumed = 0;
    imgReader.restart();

    int posCount = fillPassedSamples( 0, numPos, true, posConsumed );
    if( !posCount )
        return false;
    cout << "POS count : consumed   " << posCount << " : " << (int)posConsumed << endl;

    int proNumNeg = cvRound( ( ((double)numNeg) * ((double)posCount) ) / numPos ); // apply only a fraction of negative samples. double is required since overflow is possible
    int negCount = fillPassedSamples( posCount, proNumNeg, false, negConsumed );
    if ( !negCount )
        return false;

    curNumSamples = posCount + negCount;
    acceptanceRatio = negConsumed == 0 ? 0 : ( (double)negCount/(double)(int64)negConsumed );
    cout << "NEG count : acceptanceRatio    " << negCount << " : " << acceptanceRatio << endl;
    return true;
}

int BrCascadeClassifier::fillPassedSamples( int first, int count, bool isPositive, int64& consumed )
{
    int getcount = 0;
    Mat img(cascadeParams.winSize, CV_8UC1);
    for( int i = first; i < first + count; i++ )
    {
        for( ; ; )
        {
            bool isGetImg = isPositive ? imgReader.getPos( img ) :
                                           imgReader.getNeg( img );
            if( !isGetImg )
                return getcount;
            consumed++;

            featureEvaluator->setImage( img, isPositive ? 1 : 0, i );
            if( predict( i ) == 1.0F )
            {
                getcount++;
                printf("%s current samples: %d\r", isPositive ? "POS":"NEG", getcount);
                break;
            }
        }
    }
    return getcount;
}

void BrCascadeClassifier::writeParams( FileStorage &fs ) const
{
    cascadeParams.write( fs );
    fs << CC_STAGE_PARAMS << "{"; stageParams->write( fs ); fs << "}";
    fs << CC_FEATURE_PARAMS << "{"; featureParams->write( fs ); fs << "}";
}

void BrCascadeClassifier::writeFeatures( FileStorage &fs, const Mat& featureMap ) const
{
    ((FeatureEvaluator*)((Ptr<FeatureEvaluator>)featureEvaluator))->writeFeatures( fs, featureMap );
}

void BrCascadeClassifier::writeStages( FileStorage &fs, const Mat& featureMap ) const
{
    char cmnt[30];
    int i = 0;
    fs << CC_STAGES << "[";
    for( vector< Ptr<CascadeBoost> >::const_iterator it = stageClassifiers.begin();
        it != stageClassifiers.end(); it++, i++ )
    {
        sprintf( cmnt, "stage %d", i );
        cvWriteComment( fs.fs, cmnt, 0 );
        fs << "{";
        ((CascadeBoost*)((Ptr<CascadeBoost>)*it))->write( fs, featureMap );
        fs << "}";
    }
    fs << "]";
}

bool BrCascadeClassifier::readParams( const FileNode &node )
{
    if ( !node.isMap() || !cascadeParams.read( node ) )
        return false;
    stageParams = new CascadeBoostParams;
    FileNode rnode = node[CC_STAGE_PARAMS];
    if ( !stageParams->read( rnode ) )
        return false;

    featureParams = FeatureParams::create(cascadeParams.featureType);
    rnode = node[CC_FEATURE_PARAMS];
    if ( !featureParams->read( rnode ) )
        return false;
    return true;
}

bool BrCascadeClassifier::readStages( const FileNode &node)
{
    FileNode rnode = node[CC_STAGES];
    if (!rnode.empty() || !rnode.isSeq())
        return false;
    stageClassifiers.reserve(numStages);
    FileNodeIterator it = rnode.begin();
    for( int i = 0; i < min( (int)rnode.size(), numStages ); i++, it++ )
    {
        CascadeBoost* tempStage = new CascadeBoost;
        if ( !tempStage->read( *it, (FeatureEvaluator *)featureEvaluator, *((CascadeBoostParams*)stageParams) ) )
        {
            delete tempStage;
            return false;
        }
        stageClassifiers.push_back(tempStage);
    }
    return true;
}

void BrCascadeClassifier::save( const string filename, bool baseFormat )
{
    FileStorage fs( filename, FileStorage::WRITE );

    if ( !fs.isOpened() )
        return;

    fs << FileStorage::getDefaultObjectName(filename) << "{";
    if ( !baseFormat )
    {
        Mat featureMap;
        getUsedFeaturesIdxMap( featureMap );
        writeParams( fs );
        fs << CC_STAGE_NUM << (int)stageClassifiers.size();
        writeStages( fs, featureMap );
        writeFeatures( fs, featureMap );
    }
    else
    {
        qFatal("Old style cascade. Not sure how you got here but it's not supported");
    }
    fs << "}";
}

bool BrCascadeClassifier::load( const string cascadeDirName )
{
    FileStorage fs( cascadeDirName + CC_PARAMS_FILENAME, FileStorage::READ );
    if ( !fs.isOpened() )
        return false;
    FileNode node = fs.getFirstTopLevelNode();
    if ( !readParams( node ) )
        return false;

    featureEvaluator = FeatureEvaluator::create(cascadeParams.featureType);
    featureEvaluator->init( ((FeatureParams*)featureParams), numPos + numNeg, cascadeParams.winSize );
    fs.release();

    char buf[10];
    for ( int si = 0; si < numStages; si++ )
    {
        sprintf( buf, "%s%d", "stage", si);
        fs.open( cascadeDirName + buf + ".xml", FileStorage::READ );
        node = fs.getFirstTopLevelNode();
        if ( !fs.isOpened() )
            break;
        CascadeBoost *tempStage = new CascadeBoost;

        if ( !tempStage->read( node, (FeatureEvaluator*)featureEvaluator, *((CascadeBoostParams*)stageParams )) )
        {
            delete tempStage;
            fs.release();
            break;
        }
        stageClassifiers.push_back(tempStage);
    }
    return true;
}

void BrCascadeClassifier::getUsedFeaturesIdxMap( Mat& featureMap )
{
    int varCount = featureEvaluator->getNumFeatures() * featureEvaluator->getFeatureSize();
    featureMap.create( 1, varCount, CV_32SC1 );
    featureMap.setTo(Scalar(-1));

    for( vector< Ptr<CascadeBoost> >::const_iterator it = stageClassifiers.begin();
        it != stageClassifiers.end(); it++ )
        ((CascadeBoost*)((Ptr<CascadeBoost>)(*it)))->markUsedFeaturesInMap( featureMap );

    for( int fi = 0, idx = 0; fi < varCount; fi++ )
        if ( featureMap.at<int>(0, fi) >= 0 )
            featureMap.ptr<int>(0)[fi] = idx++;
}

