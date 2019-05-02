#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \author Scott Klum \cite sklum
 * \brief Document Me!
 */
class IfTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(br::Transform* transform READ get_transform WRITE set_transform RESET reset_transform)
    Q_PROPERTY(QString key READ get_key WRITE set_key RESET reset_key)
    Q_PROPERTY(QString value READ get_value WRITE set_value RESET reset_value)
    Q_PROPERTY(QString comparison READ get_comparison WRITE set_comparison RESET reset_comparison)
    Q_PROPERTY(bool projectOnEmpty READ get_projectOnEmpty WRITE set_projectOnEmpty RESET reset_projectOnEmpty)
    BR_PROPERTY(br::Transform*, transform, NULL)
    BR_PROPERTY(QString, key, "Label")
    BR_PROPERTY(QString, value, "1")
    BR_PROPERTY(QString, comparison, "e")
    BR_PROPERTY(bool, projectOnEmpty, false)

    bool compare(const QVariant &metadata) const
    {
        bool result, ok1 = true, ok2 = true;

        if (comparison == "empty")
	    result = metadata.isNull() || (metadata.canConvert<QString>() && metadata.toString().isEmpty());
        else if (comparison == "e")
            result = metadata == value;
        else if (comparison == "ne")
            result = metadata != value;
        else if (comparison == "l")
            result = metadata.toFloat(&ok1) < value.toFloat(&ok2);
        else if (comparison == "g")
            result = metadata.toFloat(&ok1) > value.toFloat(&ok2);
        else if (comparison == "le")
            result = metadata.toFloat(&ok1) <= value.toFloat(&ok2);
        else if (comparison == "ge")
            result = metadata.toFloat(&ok1) >= value.toFloat(&ok2);
        else if (comparison == "c") // contains
            result = metadata.toString().contains(value);
        else
            qFatal("Unrecognized comparison string.");

        if (!(ok1 && ok2))
            qFatal("Cannot convert metadata value and/or comparison to float.");

        return result;
    }

public:
    void train(const TemplateList &data)
    {
        if (transform->trainable) {
            TemplateList passed;
            for (int i=0; i<data.size(); i++)
                if ((data[i].file.contains(key) || projectOnEmpty) && compare(data[i].file.value(key)))
                    passed.append(data[i]);
            transform->train(passed);
        }
    }

    void project(const Template &src, Template &dst) const {
        if ((src.file.contains(key) || projectOnEmpty) && compare(src.file.value(key)))
            transform->project(src, dst);
        else
            dst = src;
    }

    void project(const TemplateList &src, TemplateList &dst) const {
        TemplateList ifTrue, ifFalse;
        foreach (const Template &t, src) {
            if ((t.file.contains(key) || projectOnEmpty) && compare(t.file.value(key)))
                ifTrue.append(t);
            else
                ifFalse.append(t);
        }

        transform->project(ifTrue, dst);
        dst.append(ifFalse);
    }

};
BR_REGISTER(Transform, IfTransform)

} // namespace br

#include "imgproc/if.moc"
