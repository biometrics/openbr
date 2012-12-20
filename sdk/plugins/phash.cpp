#include <opencv2/core/core.hpp>
#include <pHash.h>
#include <mm_plugin.h>

using namespace mm;

/**** PHASH ****/
class pHashEnroll : public UntrainableFeature
{
    void project(const Matrix &src, Matrix &dst) const
    {
        CImg<uint8_t> cImg(src.data, src.cols, src.rows, 1, src.channels());
        cv::Mat m(1, sizeof(ulong64), CV_8UC1);
        ulong64 hash;
        if (ph_dct_imagehash(cImg, hash) == -1)
            qFatal("pHashEnroll::project ph_dct_imagehash failure for file %s.", qPrintable(src.metadata.fileName));
        memcpy(m.data, &hash, sizeof(ulong64));
        dst = Matrix(m, src.metadata);
    }

    /*** Taken from pHash, modified to take in a CImg instead of a file. ***/
    static CImg<float>* ph_dct_matrix(const int N){
        CImg<float> *ptr_matrix = new CImg<float>(N,N,1,1,1/sqrt((float)N));
        const float c1 = sqrt(2.0/N);
        for (int x=0;x<N;x++){
            for (int y=1;y<N;y++){
                *ptr_matrix->data(x,y) = c1*cos((cimg::PI/2/N)*y*(2*x+1));
            }
        }
        return ptr_matrix;
    }

    static int ph_dct_imagehash(CImg<uint8_t> src, ulong64 &hash)
    {
        CImg<float> meanfilter(7,7,1,1,1);
        CImg<float> img;
        if (src.spectrum() == 3){
            img = src.RGBtoYCbCr().channel(0).get_convolve(meanfilter);
        } else if (src.spectrum() == 4){
            int width = img.width();
            int height = img.height();
            int depth = img.depth();
            img = src.crop(0,0,0,0,width-1,height-1,depth-1,2).RGBtoYCbCr().channel(0).get_convolve(meanfilter);
        } else {
            img = src.channel(0).get_convolve(meanfilter);
        }

        img.resize(32,32);
        CImg<float> *C  = ph_dct_matrix(32);
        CImg<float> Ctransp = C->get_transpose();

        CImg<float> dctImage = (*C)*img*Ctransp;

        CImg<float> subsec = dctImage.crop(1,1,8,8).unroll('x');;

        float median = subsec.median();
        ulong64 one = 0x0000000000000001;
        hash = 0x0000000000000000;
        for (int i=0;i< 64;i++){
            float current = subsec(i);
            if (current > median)
                hash |= one;
            one = one << 1;
        }

        delete C;

        return 0;
    }
};

MM_REGISTER(Feature, pHashEnroll, false)


/**** PHASH_COMPARE ****/
class pHashCompare : public ComparerBase
{
    float compare(const cv::Mat &a, const cv::Mat &b) const
    {
        return 1.f - 1.f * ph_hamming_distance(*reinterpret_cast<ulong64*>(a.data), *reinterpret_cast<ulong64*>(b.data)) / 64;
    }
};

MM_REGISTER(Comparer, pHashCompare, false)


/**** PHASH ****/
class pHash : public Algorithm
{
    QString algorithm() const
    {
        return "Open+pHashEnroll:Identity:pHashCompare";
    }
};

MM_REGISTER(Algorithm, pHash, false)
