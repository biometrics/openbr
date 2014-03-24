#include <likely.h>
#include <likely/opencv.hpp>

#include "openbr_internal.h"

namespace br
{

/*!
 * \ingroup transforms
 * \brief Generic interface to Likely JIT compiler
 *
 * www.liblikely.org
 * \author Josh Klontz \cite jklontz
 */
class LikelyTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(QString kernel READ get_kernel WRITE set_kernel RESET reset_kernel STORED false)
    BR_PROPERTY(QString, kernel, "")

    likely_function function;

    void init()
    {
        likely_ast ast = likely_ast_from_string(qPrintable(kernel));
        likely_env env = likely_new_env();
        function = likely_compile(ast, env, likely_type_null);
        likely_release_env(env);
        likely_release_ast(ast);
    }

    void project(const Template &src, Template &dst) const
    {
        likely_const_mat srcl = likely::fromCvMat(src);
        likely_const_mat dstl = function(srcl);
        dst = likely::toCvMat(dstl);
        likely_release(dstl);
        likely_release(srcl);
    }
};

BR_REGISTER(Transform, LikelyTransform)

/*!
 * \ingroup formats
 * \brief Likely matrix format
 *
 * www.liblikely.org
 * \author Josh Klontz \cite jklontz
 */
class lmFormat : public Format
{
    Q_OBJECT

    Template read() const
    {
        likely_const_mat m = likely_read(qPrintable(file.name));
        Template result(likely::toCvMat(m));
        likely_release(m);
        return result;
    }

    void write(const Template &t) const
    {
        likely_const_mat m = likely::fromCvMat(t);
        likely_write(m, qPrintable(file.name));
        likely_release(m);
    }
};

BR_REGISTER(Format, lmFormat)

} // namespace br

#include "likely.moc"
