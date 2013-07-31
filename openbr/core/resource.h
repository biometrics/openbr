/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2012 The MITRE Corporation                                      *
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License");           *
 * you may not use this file except in compliance with the License.          *
 * You may obtain a copy of the License at                                   *
 *                                                                           *
 *     http://www.apache.org/licenses/LICENSE-2.0                            *
 *                                                                           *
 * Unless required by applicable law or agreed to in writing, software       *
 * distributed under the License is distributed on an "AS IS" BASIS,         *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 * See the License for the specific language governing permissions and       *
 * limitations under the License.                                            *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef __RESOURCE_H
#define __RESOURCE_H

#include <QHash>
#include <QList>
#include <QMutex>
#include <QSemaphore>
#include <QSharedPointer>
#include <QString>
#include <QThread>
#include <openbr/openbr_plugin.h>

template <typename T>
class ResourceMaker
{
public:
    virtual ~ResourceMaker() {}
    virtual T *make() const = 0;
};

template <typename T>
class DefaultResourceMaker : public ResourceMaker<T>
{
    T *make() const { return new T(); }
};

template <typename T>
class Resource
{
    QSharedPointer< ResourceMaker<T> > resourceMaker;
    QSharedPointer< QList<T*> > availableResources;
    QSharedPointer<QMutex> lock;
    QSharedPointer<QSemaphore> totalResources;

public:
    Resource(ResourceMaker<T> *rm = new DefaultResourceMaker<T>())
        : resourceMaker(rm)
        , availableResources(new QList<T*>())
        , lock(new QMutex())
        , totalResources(new QSemaphore(br::Globals->parallelism))
    {}

    ~Resource()
    {
        qDeleteAll(*availableResources);
    }

    T *acquire() const
    {
        totalResources->acquire();
        lock->lock();

        if (availableResources->isEmpty())
            availableResources->append(resourceMaker->make());
        T* resource = availableResources->takeFirst();

        lock->unlock();

        return resource;
    }

    void release(T *resource) const
    {
        lock->lock();
        availableResources->append(resource);
        lock->unlock();
        totalResources->release();
    }

    void setResourceMaker(ResourceMaker<T> *maker)
    {
        resourceMaker = QSharedPointer< ResourceMaker<T> >(maker);
    }

    void setMaxResources(int max)
    {
        totalResources = QSharedPointer<QSemaphore>(new QSemaphore(max));
    }
};

#endif //__RESOURCE_H
