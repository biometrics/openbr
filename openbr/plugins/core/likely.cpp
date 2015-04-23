#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Generic interface to Likely JIT compiler
 *
 * \br_link www.liblikely.org
 * \author Josh Klontz \cite jklontz
 */
class LikelyTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(QString kernel READ get_kernel WRITE set_kernel RESET reset_kernel STORED false)
    BR_PROPERTY(QString, kernel, "")

    likely_const_env env;
    void *function;

    ~LikelyTransform()
    {
        likely_release_env(env);
    }

    void init()
    {
        likely_release_env(env);
        const likely_ast ast = likely_lex_and_parse(qPrintable(kernel), likely_file_lisp);
        const likely_const_env parent = likely_standard(NULL);
        env = likely_eval(ast, parent, NULL, NULL);
        likely_release_env(parent);
        likely_release_ast(ast);
        function = likely_compile(env->expr, NULL, 0);
    }

    void project(const Template &src, Template &dst) const
    {
        const likely_const_mat srcl = likelyFromOpenCVMat(src);
        const likely_const_mat dstl = reinterpret_cast<likely_mat (*)(likely_const_mat)>(function)(srcl);
        dst = likelyToOpenCVMat(dstl);
        likely_release_mat(dstl);
        likely_release_mat(srcl);
    }

public:
    LikelyTransform()
    {
        env = NULL;
    }
};

BR_REGISTER(Transform, LikelyTransform)

} // namespace br

#include "core/likely.moc"
