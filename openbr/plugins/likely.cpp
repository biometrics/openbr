#include <likely.h>

#include "openbr_internal.h"

namespace br
{

class LikelyTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(QString kernel READ get_kernel WRITE set_kernel RESET reset_kernel STORED false)
    BR_PROPERTY(QString, kernel, "")

    likely_function function;

    void init()
    {
        likely_ast ast = likely_asts_from_string(qPrintable(kernel));
        likely_env env = likely_new_env();
        function = likely_compile(ast, env, likely_type_null);
        likely_release_env(env);
        likely_release_ast(ast);
    }

    void project(const Template &src, Template &dst) const
    {
        dst = src;
    }
};

BR_REGISTER(Transform, LikelyTransform)

}

#include "likely.moc"
