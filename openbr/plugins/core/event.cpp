#include <openbr/plugins/openbr_internal.h>

namespace br
{

class EventTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(QString eventName READ get_eventName WRITE set_eventName RESET reset_eventName STORED false)
    BR_PROPERTY(QString, eventName, "")

    TemplateEvent event;

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        event.pulseSignal(dst);
    }

    TemplateEvent *getEvent(const QString &name)
    {
        return name == eventName ? &event : NULL;
    }
};

BR_REGISTER(Transform, EventTransform)

} // namespace br

#include "core/event.moc"
