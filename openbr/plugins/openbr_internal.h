#ifndef __OPENBR_INTERNAL_H
#define __OPENBR_INTERNAL_H

#include "openbr/openbr_plugin.h"
#include "openbr/core/resource.h"

namespace br
{
/*!
 * \brief A br::Transform that does not require training data.
 */
class BR_EXPORT UntrainableTransform : public Transform
{
    Q_OBJECT

protected:
    UntrainableTransform(bool independent = true) : Transform(independent, false) {} /*!< \brief Construct an untrainable transform. */

private:
    Transform *clone() const { return const_cast<UntrainableTransform*>(this); }
    void train(const TemplateList &data) { (void) data; }
    void store(QDataStream &stream) const { (void) stream; }
    void load(QDataStream &stream) { (void) stream; }
};

/*!
 * \brief A br::MetaTransform that does not require training data.
 */
class BR_EXPORT UntrainableMetaTransform : public UntrainableTransform
{
    Q_OBJECT

protected:
    UntrainableMetaTransform() : UntrainableTransform(false) {}
};

/*!
 * \brief A br::Transform for which the results of project may change due to prior calls to project
 */
class BR_EXPORT TimeVaryingTransform : public Transform
{
    Q_OBJECT

public:
    virtual bool timeVarying() const { return true; }

    virtual void project(const Template &src, Template &dst) const
    {
        qFatal("No const project defined for time-varying transform");
        (void) dst; (void) src;
    }

    virtual void project(const TemplateList &src, TemplateList &dst) const
    {
        qFatal("No const project defined for time-varying transform");
        (void) dst; (void) src;
    }

    // Get a compile failure if this isn't here to go along with the other
    // projectUpdate, no idea why
    virtual void projectUpdate(const Template & src, Template & dst)
    {
        (void) src; (void) dst;
        qFatal("do something useful");
    }

    virtual void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        foreach (const Template & src_part, src) {
            Template out;
            projectUpdate(src_part, out);
            dst.append(out);
        }
    }

    /*!
     *\brief For transforms that don't do any training, this default implementation
     * which creates a new copy of the Transform from its description string is sufficient.
     */
    virtual Transform * smartCopy()
    {
        return this->clone();
    }


protected:
    TimeVaryingTransform(bool independent = true, bool trainable = true) : Transform(independent, trainable) {}
};

/*!
 * \brief A br::Transform expecting multiple matrices per template.
 */
class BR_EXPORT MetaTransform : public Transform
{
    Q_OBJECT

protected:
    MetaTransform() : Transform(false) {}
};


class TransformCopier : public ResourceMaker<Transform>
{
public:
    Transform * basis;
    TransformCopier(Transform * _basis)
    {
        basis = _basis;
    }

    virtual Transform *make() const
    {
        return basis->smartCopy();
    }

};

class TimeInvariantWrapperTransform : public MetaTransform
{
public:
    Resource<Transform> transformSource;

    TimeInvariantWrapperTransform(Transform * basis) : transformSource(new TransformCopier(basis))
    {
        baseTransform = basis;
    }

    virtual void project(const Template &src, Template &dst) const
    {
        Transform * aTransform = transformSource.acquire();
        aTransform->projectUpdate(src,dst);
        transformSource.release(aTransform);
    }


    void project(const TemplateList &src, TemplateList &dst) const
    {
        Transform * aTransform = transformSource.acquire();
        aTransform->projectUpdate(src,dst);
        transformSource.release(aTransform);
    }

    void train(const TemplateList &data)
    {
        baseTransform->train(data);
    }

private:
    Transform * baseTransform;
};


/*!
 * \brief A MetaTransform that aggregates some sub-transforms
 */
class BR_EXPORT CompositeTransform : public TimeVaryingTransform
{
    Q_OBJECT

public:
    Q_PROPERTY(QList<br::Transform*> transforms READ get_transforms WRITE set_transforms RESET reset_transforms)
    BR_PROPERTY(QList<br::Transform*>, transforms, QList<br::Transform*>())

    virtual void project(const Template &src, Template &dst) const
    {
        if (timeVarying()) qFatal("No const project defined for time-varying transform");
        _project(src, dst);
    }

    virtual void project(const TemplateList &src, TemplateList &dst) const
    {
        if (timeVarying()) {
            if (!this->timeInvariantAlias) {
                QMutexLocker lock(&aliasLock);
                CompositeTransform * non_const = const_cast<CompositeTransform *>(this);
                non_const->timeInvariantAlias = non_const->smartCopy();
                non_const->timeInvariantAlias->setParent(non_const);
                lock.unlock();
            }
            timeInvariantAlias->projectUpdate(src,dst);
            return;
        }
        _project(src, dst);
    }


    bool timeVarying() const { return isTimeVarying; }

    void init()
    {
        isTimeVarying = false;
        trainable = false;
        foreach (const br::Transform *transform, transforms) {
            isTimeVarying = isTimeVarying || transform->timeVarying();
            trainable = trainable || transform->trainable;
        }
    }

    /*!
     * \brief Composite transforms need to create a copy of themselves if they
     * have any time-varying children. If this object is flagged as time-varying,
     * it creates a new copy of its own class, and gives that copy the child transforms
     * returned by calling smartCopy on this transforms children
     */
    Transform * smartCopy()
    {
        if (!timeVarying())
            return this;

        QString name = metaObject()->className();
        name.replace("Transform","");
        name += "([])";
        name.replace("br::","");
        CompositeTransform * output = dynamic_cast<CompositeTransform *>(Transform::make(name, NULL));

        if (output == NULL)
            qFatal("Dynamic cast failed!");

        foreach(Transform* t, transforms )
        {
            Transform * maybe_copy = t->smartCopy();
            if (maybe_copy->parent() == NULL)
                maybe_copy->setParent(output);
            output->transforms.append(t->smartCopy());
        }

        output->file = this->file;
        output->classes = classes;
        output->instances = instances;
        output->fraction = fraction;

        output->init();

        return output;
    }

protected:
    bool isTimeVarying;

    mutable QMutex aliasLock;
    Transform * timeInvariantAlias;

    virtual void _project(const Template & src, Template & dst) const = 0;
    virtual void _project(const TemplateList & src, TemplateList & dst) const = 0;

    CompositeTransform() : TimeVaryingTransform(false) { timeInvariantAlias = NULL; }
};

}

#endif
