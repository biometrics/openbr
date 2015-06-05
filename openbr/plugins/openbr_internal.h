#ifndef OPENBR_INTERNAL_H
#define OPENBR_INTERNAL_H

#include "openbr/openbr_plugin.h"
#include "openbr/core/resource.h"

namespace br
{

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

class BR_EXPORT MetaTransform : public Transform
{
    Q_OBJECT

protected:
    MetaTransform() : Transform(false) {}
};

class BR_EXPORT UntrainableMetaTransform : public UntrainableTransform
{
    Q_OBJECT

protected:
    UntrainableMetaTransform() : UntrainableTransform(false) {}
};

class TransformCopier : public ResourceMaker<Transform>
{
public:
    Transform *basis;
    TransformCopier(Transform *_basis)
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

    TimeInvariantWrapperTransform(Transform *basis) : transformSource(new TransformCopier(basis))
    {
        if (!basis)
            qFatal("TimeInvariantWrapper created with NULL transform");
        baseTransform = basis;
        trainable = basis->trainable;
    }

    virtual void project(const Template &src, Template &dst) const
    {
        Transform *aTransform = transformSource.acquire();
        aTransform->projectUpdate(src,dst);
        transformSource.release(aTransform);
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        Transform *aTransform = transformSource.acquire();
        aTransform->projectUpdate(src,dst);
        transformSource.release(aTransform);
    }

    void train(const QList<TemplateList> &data)
    {
        baseTransform->train(data);
    }

private:
    Transform *baseTransform;
};

class BR_EXPORT TimeVaryingTransform : public Transform
{
    Q_OBJECT

public:

    virtual bool timeVarying() const { return true; }

    virtual void project(const Template &src, Template &dst) const
    {
        timeInvariantAlias.project(src,dst);
    }

    virtual void project(const TemplateList &src, TemplateList &dst) const
    {
        timeInvariantAlias.project(src,dst);
    }

    // Get a compile failure if this isn't here to go along with the other
    // projectUpdate, no idea why
    virtual void projectUpdate(const Template &src, Template &dst)
    {
        (void) src; (void) dst;
        qFatal("do something useful");
    }

    virtual void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        foreach (const Template &src_part, src) {
            Template out;
            projectUpdate(src_part, out);
            dst.append(out);
        }
    }

    virtual Transform *smartCopy(bool &newTransform)
    {
        newTransform = true;
        return this->clone();
    }

protected:
    // Since copies aren't actually made until project is called, we can set up
    // timeInvariantAlias in the constructor.
    TimeInvariantWrapperTransform timeInvariantAlias;
    TimeVaryingTransform(bool independent = true, bool trainable = true) : Transform(independent, trainable), timeInvariantAlias(this)
    {
        //
    }
};

class BR_EXPORT WrapperTransform : public TimeVaryingTransform
{
    Q_OBJECT
public:
    WrapperTransform(bool independent = true) : TimeVaryingTransform(independent)
    {
    }

    Q_PROPERTY(br::Transform *transform READ get_transform WRITE set_transform RESET reset_transform STORED false)
    BR_PROPERTY(br::Transform *, transform, NULL)

    bool timeVarying() const { return transform->timeVarying(); }

    void project(const Template &src, Template &dst) const
    {
        transform->project(src,dst);
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        transform->project(src, dst);
    }

    void projectUpdate(const Template &src, Template &dst)
    {
        transform->projectUpdate(src,dst);
    }
    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        transform->projectUpdate(src,dst);
    }

    void train(const QList<TemplateList> &data)
    {
        transform->train(data);
    }

    virtual void finalize(TemplateList &output)
    {
        transform->finalize(output);
    }

    void init()
    {
        if (transform)
            this->trainable = transform->trainable;
    }

    virtual Transform *simplify(bool &newTransform)
    {
        newTransform = false;
        bool newChild = false;
        Transform *temp = transform->simplify(newTransform);
        if (temp == transform)
            return this;

        if (!temp)
            return NULL;

        // else make a copy to point at the new transform
        Transform *child = transform;
        transform = NULL;
        WrapperTransform *output = dynamic_cast<WrapperTransform *>(Transform::make(description(), NULL));
        transform = child;

        output->transform = temp;

        if (newChild)
            temp->setParent(output);

        newTransform = true;
        return output;
    }

    Transform *smartCopy(bool &newTransform)
    {
        if (!timeVarying()) {
            newTransform = false;
            return this;
        }
        newTransform = true;
        Transform *temp = transform;
        transform = NULL;
        WrapperTransform *output = dynamic_cast<WrapperTransform *>(Transform::make(description(), NULL));
        transform = temp;

        if (output == NULL)
            qFatal("Dynamic cast failed!");

        bool newItem = false;
        Transform *maybe_copy = transform->smartCopy(newItem);
        if (newItem)
            maybe_copy->setParent(output);
        output->transform = maybe_copy;

        output->file = this->file;
        output->init();

        return output;
    }

};

class BR_EXPORT CompositeTransform : public TimeVaryingTransform
{
    Q_OBJECT

public:
    Q_PROPERTY(QList<br::Transform*> transforms READ get_transforms WRITE set_transforms RESET reset_transforms)
    BR_PROPERTY(QList<br::Transform*>, transforms, QList<br::Transform*>())

    virtual void project(const Template &src, Template &dst) const
    {
        if (timeVarying()) {
            timeInvariantAlias.project(src,dst);
            return;
        }
        _project(src, dst);
    }

    virtual void project(const TemplateList &src, TemplateList &dst) const
    {
        if (timeVarying()) {
            timeInvariantAlias.project(src,dst);
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

    Transform *smartCopy(bool &newTransform)
    {
        if (!timeVarying()) {
            newTransform = false;
            return this;
        }
        newTransform = true;

        QList<Transform *> temp = transforms;
        transforms = QList<Transform *>();
        CompositeTransform *output = dynamic_cast<CompositeTransform *>(Transform::make(description(), NULL));
        transforms = temp;

        if (output == NULL)
            qFatal("Dynamic cast failed!");

        foreach (Transform* t, transforms ) {
            bool newItem = false;
            Transform *maybe_copy = t->smartCopy(newItem);
            if (newItem)
                maybe_copy->setParent(output);
            output->transforms.append(maybe_copy);
        }

        output->file = this->file;
        output->init();

        return output;
    }

    virtual Transform *simplify(bool &newTransform)
    {
        newTransform = false;
        QList<Transform *> newTransforms;
        bool anyNew = false;

        QList<bool> newChildren;
        for (int i=0; i < transforms.size();i++)
        {
            bool newChild = false;
            Transform *temp = transforms[i]->simplify(newChild);
            if (temp == NULL) {
                anyNew = true;
                continue;
            }
            newTransforms.append(temp);
            newChildren.append(newChild);
            if (temp != transforms[i])
                anyNew = true;
        }

        if (newTransforms.empty() )
            return NULL;

        if (!anyNew)
            return this;

        // make a copy of the current object, with empty transforms
        QList<Transform *> children = transforms;
        transforms = QList<Transform *> ();
        CompositeTransform *output = dynamic_cast<CompositeTransform *>(Transform::make(description(false), NULL));
        transforms = children;

        output->transforms = newTransforms;
        for (int i=0;i < newChildren.size();i++)
        {
            if (newChildren[i])
                output->transforms[i]->setParent(output);
        }
        output->init();

        newTransform = true;
        return output;
    }

protected:
    bool isTimeVarying;

    virtual void _project(const Template &src, Template &dst) const = 0;
    virtual void _project(const TemplateList &src, TemplateList &dst) const = 0;

    CompositeTransform() : TimeVaryingTransform(false) {}
};

class EnrollmentWorker;

// Implemented in plugins/process.cpp
struct WorkerProcess
{
    QString transform;
    QString baseName;
    EnrollmentWorker *processInterface;

    void mainLoop();
};

class MetadataTransform : public Transform
{
    Q_OBJECT
public:

    virtual void projectMetadata(const File &src, File &dst) const = 0;

    void project(const Template &src, Template &dst) const
    {
        dst = src;
        projectMetadata(src.file, dst.file);
    }

protected:
    MetadataTransform(bool trainable = true) : Transform(false,trainable) {}
};

class UntrainableMetadataTransform : public MetadataTransform
{
    Q_OBJECT

protected:
    UntrainableMetadataTransform() : MetadataTransform(false) {}
};

class FileGallery : public Gallery
{
    Q_OBJECT
public:
    QFile f;

    virtual ~FileGallery() { f.close(); }

    void init();

    qint64 totalSize();
    qint64 position() { return f.pos(); }

    bool readOpen();
    void writeOpen();

};


void applyAdditionalProperties(const File &temp, Transform *target);


inline void splitFTEs(TemplateList &src, TemplateList  &ftes)
{
    TemplateList active = src;
    src.clear();

    foreach (const Template &t, active) {
        if (t.file.fte) {
            if (!Globals->enrollAll)
                ftes.append(t);
        } else
            src.append(t);
    }
}

typedef QPair<int,float> Neighbor; // QPair<id,similarity>
typedef QList<Neighbor> Neighbors;
typedef QVector<Neighbors> Neighborhood;

BR_EXPORT bool compareNeighbors(const Neighbor &a, const Neighbor &b);

class BR_EXPORT UntrainableDistance : public Distance
{
    Q_OBJECT

private:
    bool trainable() { return false; }
    void train(const TemplateList &data) { (void) data; }
};

/*!
 * \brief A br::Distance that checks the elements of its list property to see if it needs to be trained.
 */
class BR_EXPORT ListDistance : public Distance
{
    Q_OBJECT

public:
    Q_PROPERTY(QList<br::Distance*> distances READ get_distances WRITE set_distances RESET reset_distances)
    BR_PROPERTY(QList<br::Distance*>, distances, QList<br::Distance*>())

    bool trainable()
    {
        for (int i=0; i<distances.size(); i++)
            if (distances[i]->trainable())
                return true;
        return false;
    }
};

}

#endif // OPENBR_INTERNAL_H
