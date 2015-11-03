#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \author Brendan Klare \cite bklare
 * \brief Randomly sample templates from a gallery
 */
class RandomTemplatesTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(float percent READ get_percent WRITE set_percent RESET reset_percent)
    BR_PROPERTY(float, percent, .01)

    void project(const Template &src, Template &dst) const {
		qFatal("Not supported in RandomTemplates.");
    }

    void project(const TemplateList &src, TemplateList &dst) const {
		for (int i = 0; i < src.size(); i++) {
			const float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
			if (r <= percent)
				dst.append(src[i]);
		}
    }
};
BR_REGISTER(Transform, RandomTemplatesTransform)

} // namespace br

#include "imgproc/randomtemplates.moc"

