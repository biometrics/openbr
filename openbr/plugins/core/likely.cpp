#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/qtutils.h>

#include <likely.h>
#include <likely/opencv.hpp>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Generic interface to Likely JIT compiler
 * \br_link Homepage http://liblikely.org
 * \br_link API Documentation https://s3.amazonaws.com/liblikely/doxygen/index.html
 * \author Josh Klontz \cite jklontz
 */
class LikelyTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(QString sourceFile READ get_sourceFile WRITE set_sourceFile RESET reset_sourceFile STORED false)
    BR_PROPERTY(QString, sourceFile, "")

    QByteArray bitcode;
    likely_env env;

    typedef likely_mat (*Function)(likely_const_mat);
    Function function;

    ~LikelyTransform()
    {
        likely_release_env(env);
    }

    void compile()
    {
        const likely_const_mat data = likely_new(likely_u8 | likely_multi_channel, bitcode.size(), 1, 1, 1, bitcode.data());
        env = likely_precompiled(data, qPrintable(QFileInfo(sourceFile).baseName()));
        function = (Function) likely_function(env->expr);
        if (!function)
            qFatal("Failed to compile: %s", qPrintable(sourceFile));
    }

    void train(const TemplateList &trainingData)
    {
        const likely_type source_file_type = likely_guess_file_type(qPrintable(sourceFile));
        const likely_const_mat source_code = likely_read(qPrintable(sourceFile), source_file_type, likely_void);
        const likely_const_mat data = likelyFromOpenCVMats(trainingData.data().toVector().toStdVector());

        // Pick settings to minimize code size
        likely_settings settings = likely_default_settings(likely_file_bitcode, false);
        settings.runtime_only = true; // The compiled algorithm should not depend on any external functions,
                                      // except for those in the likely runtime API.

        // Construct a compilation environment
        likely_mat output = NULL;
        likely_const_env parent = likely_standard(settings, &output, likely_file_bitcode);

        { // Define the `data` variable
            const likely_const_env env = likely_define("data", data, parent);
            likely_release_env(parent);
            parent = env;
        }

        likely_release_env(likely_lex_parse_and_eval(source_code->data, source_file_type, parent));
        likely_release_env(parent);

        assert(output);
        bitcode = QByteArray(output->data, likely_bytes(output));
        likely_release_mat(output);

        compile();
        likely_release_mat(data);
        likely_release_mat(source_code);
    }

    void project(const Template &src, Template &dst) const
    {
        const likely_const_mat srcl = likelyFromOpenCVMat(src);
        const likely_const_mat dstl = function(srcl);
        dst = likelyToOpenCVMat(dstl);
        likely_release_mat(dstl);
        likely_release_mat(srcl);
    }

    void store(QDataStream &stream) const
    {
        stream << bitcode;
    }

    void load(QDataStream &stream)
    {
        stream >> bitcode;
        compile();
    }
};

BR_REGISTER(Transform, LikelyTransform)

} // namespace br

#include "core/likely.moc"
