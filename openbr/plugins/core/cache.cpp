#include <openbr/plugins/openbr_internal.h>

namespace br
{

/*!
 * \ingroup transforms
 * \brief Caches br::Transform::project() results.
 * \author Josh Klontz \cite jklontz
 */
class CacheTransform : public MetaTransform
{
    Q_OBJECT
    Q_PROPERTY(br::Transform* transform READ get_transform WRITE set_transform RESET reset_transform)
    BR_PROPERTY(br::Transform*, transform, NULL)

    static QHash<QString, Template> cache;
    static QMutex cacheLock;

public:
    ~CacheTransform()
    {
        if (cache.isEmpty()) return;

        // Write to cache
        QFile file("Cache");
        if (!file.open(QFile::WriteOnly))
            qFatal("Unable to open %s for writing.", qPrintable(file.fileName()));
        QDataStream stream(&file);
        stream << cache;
        file.close();
    }

private:
    void init()
    {
        if (!transform) return;

        trainable = transform->trainable;
        if (!cache.isEmpty()) return;

        // Read from cache
        QFile file("Cache");
        if (file.exists()) {
            if (!file.open(QFile::ReadOnly))
                qFatal("Unable to open %s for reading.", qPrintable(file.fileName()));
            QDataStream stream(&file);
            stream >> cache;
            file.close();
        }
    }

    void train(const QList<TemplateList> &data)
    {
        transform->train(data);
    }

    void project(const Template &src, Template &dst) const
    {
        const QString &file = src.file;
        if (cache.contains(file)) {
            dst = cache[file];
        } else {
            transform->project(src, dst);
            cacheLock.lock();
            cache[file] = dst;
            cacheLock.unlock();
        }
    }
};

QHash<QString, Template> CacheTransform::cache;
QMutex CacheTransform::cacheLock;

BR_REGISTER(Transform, CacheTransform)

} // namespace br

#include "core/cache.moc"
