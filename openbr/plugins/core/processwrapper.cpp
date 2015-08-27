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

#include <QBuffer>
#include <QCoreApplication>
#include <QLocalServer>
#include <QLocalSocket>
#include <QMutex>
#include <QProcess>
#include <QUuid>
#include <QWaitCondition>

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/opencvutils.h>

Q_DECLARE_METATYPE(QLocalSocket::LocalSocketState)

using namespace cv;

namespace br
{

class CommunicationManager : public QObject
{
    Q_OBJECT
public:
    int timeout_ms;
    QThread *basis;
    CommunicationManager()
    {
        qRegisterMetaType< QAbstractSocket::SocketState> ();
        qRegisterMetaType< QLocalSocket::LocalSocketState> ();

        timeout_ms = 30000;

        basis = new QThread;
        moveToThread(basis);
        server.moveToThread(basis);
        outbound.moveToThread(basis);

        // signals for our sever
        connect(&server, SIGNAL(newConnection()), this, SLOT(receivedConnection() ));
        connect(this, SIGNAL(pulseStartServer(QString)), this, SLOT(startServerInternal(QString)), Qt::BlockingQueuedConnection);
        connect(this, SIGNAL(pulseOutboundConnect(QString)), this, SLOT(startConnectInternal(QString) ), Qt::BlockingQueuedConnection);


        // internals, cause work to be done by the main thread because reasons.
        connect(this, SIGNAL(pulseSignal()), this, SLOT(sendSignalInternal()), Qt::BlockingQueuedConnection);
        connect(this, SIGNAL(pulseReadSignal() ), this, SLOT(readSignalInternal()), Qt::BlockingQueuedConnection);
        connect(this, SIGNAL(pulseReadSerialized() ), this, SLOT(readSerializedInternal()), Qt::BlockingQueuedConnection); 
        connect(this, SIGNAL(pulseSendSerialized() ), this, SLOT(sendSerializedInternal() ), Qt::BlockingQueuedConnection);
        connect(this, SIGNAL(pulseShutdown() ), this, SLOT(shutdownInternal() ), Qt::BlockingQueuedConnection);

        // signals for our outbound connection
        connect(&outbound, SIGNAL(connected() ), this, SLOT(outboundConnected() ));
        connect(&outbound, SIGNAL(disconnected() ), this, SLOT(outboundDisconnected() ));

        connect(&outbound, SIGNAL(error(QLocalSocket::LocalSocketError)), this, SLOT(outboundConnectionError(QLocalSocket::LocalSocketError) ) );
        connect(&outbound, SIGNAL(stateChanged(QLocalSocket::LocalSocketState)), this, SLOT(outboundStateChanged(QLocalSocket::LocalSocketState) ) );

        inbound = NULL;
        basis->start();
    }

    ~CommunicationManager()
    {

    }

    void shutDownThread()
    {
        basis->quit();
        basis->wait();
        delete basis;
    }

    enum SignalType
    {
        INPUT_AVAILABLE,
        OUTPUT_AVAILABLE,
        SHOULD_END
    };


public slots:
    // matching server signals
    void receivedConnection()
    {
        inbound = server.nextPendingConnection();
        connect(inbound, SIGNAL(disconnected() ), this, SLOT(inboundDisconnected() ));
        connect(inbound, SIGNAL(error(QLocalSocket::LocalSocketError)), this, SLOT(inboundConnectionError(QLocalSocket::LocalSocketError) ) );
        connect(inbound, SIGNAL(stateChanged(QLocalSocket::LocalSocketState)), this, SLOT(inboundStateChanged(QLocalSocket::LocalSocketState) ) );

        receivedWait.wakeAll();
    }

    // matching outbound socket signals

    // Oh boy.
    void outboundConnected()
    {
        outboundWait.wakeAll();
    }

    // Oh no!
    void outboundDisconnected()
    {
        //qDebug() << key << " outbound socket has disconnected";
    }

    // informative.
    void outboundConnectionError(QLocalSocket::LocalSocketError socketError)
    {
        (void) socketError;
        //qDebug() << key << " outbound socket error " << socketError;
    }

    void outboundStateChanged(QLocalSocket::LocalSocketState socketState)
    {
        (void) socketState;
        //qDebug() << key << " outbound socket state changed to " << socketState;
    }

    // matching inbound socket signals
    void inboundDisconnected()
    {
        //qDebug() << key << " inbound socket has disconnected";
    }

    void inboundConnectionError(QLocalSocket::LocalSocketError socketError)
    {
        (void) socketError;
        //qDebug() << key << " inbound socket error " << socketError;
    }

    void inboundStateChanged(QLocalSocket::LocalSocketState socketState)
    {
        (void) socketState;
        //qDebug() << key << " inbound socket state changed to " << socketState;
    }

    void startServerInternal(QString serverName)
    {
         if (!server.isListening()) {
            bool listen_res = server.listen(serverName);
            if (!listen_res)
                qDebug() << key << " Failed to start server at " << serverName;
         }
    }

    void sendSignalInternal()
    {
        SignalType signal = sendType;
        qint64 signal_wrote = outbound.write((char *) &signal, sizeof(signal));

        if (signal_wrote != sizeof(signal))
            qDebug() << key << " inconsistent signal size";

        while (outbound.bytesToWrite() > 0) {
            bool written = outbound.waitForBytesWritten(timeout_ms);

            if (!written && (!outbound.isValid() || outbound.state() == QLocalSocket::UnconnectedState)) {
                qDebug() << key << " failed to wait for bytes written in signal size";
                return;
            }
        }
    }

    void readSignalInternal()
    {
        while (inbound->bytesAvailable() < qint64(sizeof(readSignal)) ) {
            bool size_ready = inbound->waitForReadyRead(timeout_ms);

            // wait timed out, now what?
            if (!size_ready && (!inbound->isValid() || inbound->state() == QLocalSocket::UnconnectedState)) {
                    readSignal = SHOULD_END;
                    server.close();
                    outbound.abort();
                    inbound->abort();
                    return;
            }
        }

        qint64 signalBytesRead = inbound->read((char *) &readSignal, sizeof(readSignal));
        if (signalBytesRead != sizeof(readSignal))
            qDebug("Inconsistent signal size read!");

        if (readSignal == SHOULD_END)
        {
            server.close();
            outbound.abort();
            inbound->abort();
        }
    }

    void sendSerializedInternal()
    {
        qint64 serializedSize = writeArray.size();
        qint64 size_wrote = outbound.write((char *) &serializedSize, sizeof(serializedSize));
        if (size_wrote != sizeof(serializedSize)) {
            qDebug() << key << "inconsistent size sent in send data!";
            return;
        }

        while (outbound.bytesToWrite() > 0) {
            bool written = outbound.waitForBytesWritten(timeout_ms);
        
            if (!written && (!outbound.isValid() || outbound.state() == QLocalSocket::UnconnectedState)) {
                qDebug() << key << " wait for bytes failed!";
                return;
            }
        }

        qint64 data_wrote = outbound.write(writeArray.data(), serializedSize);
        if (data_wrote != serializedSize)
            qDebug() << key << " inconsistent data written!";

        while (outbound.bytesToWrite() > 0) {
            bool write_res = outbound.waitForBytesWritten();
            if (!write_res && (!outbound.isValid() || outbound.state() == QLocalSocket::UnconnectedState)) {
                qDebug() << key << " wait for bytes failed!";
                return;
            }
        }

        return;

    }

    void readSerializedInternal()
    {
        qint64 bufferSize;
        while (inbound->bytesAvailable() < qint64(sizeof(bufferSize))) {
            bool size_ready = inbound->waitForReadyRead(timeout_ms);
            if (!size_ready && (!inbound->isValid() || inbound->state() == QLocalSocket::UnconnectedState)) {
                    qDebug() << key << " Failed to received object size in read data!";
                    qDebug() << key << "inbound status: " << inbound->state() << " error: " << inbound->errorString();
                    return;
            }
        }

        qint64 sizeBytesRead = inbound->read((char *) &bufferSize, sizeof(bufferSize));
        if (sizeBytesRead != sizeof(bufferSize)) {
            qDebug("failed to read size of buffer!");
            return;
        }

        // allocate the input buffer
        readArray.resize(bufferSize);

        // read the data, we may get it in serveral bursts
        qint64 arrayPosition = 0;
        while (arrayPosition < bufferSize) {
            while (!inbound->bytesAvailable()) {
                bool ready_res = inbound->waitForReadyRead(timeout_ms);

                if (!ready_res && (!inbound->isValid() || inbound->state() == QLocalSocket::UnconnectedState)) {
                    qDebug() << key << "failed to wait for data!";
                    return;
                }
            }

            // how many bytes do we still need?
            qint64 bytes_remaining = bufferSize - arrayPosition;
            arrayPosition += inbound->read(readArray.data()+arrayPosition, qMin(inbound->bytesAvailable(), bytes_remaining));
        }
        if (arrayPosition != bufferSize)
            qDebug() << key <<  "Read wrong size object!";
    }

    void shutdownInternal()
    {
        outbound.abort();
        inbound->abort();
        server.close();
    }


    void startConnectInternal(QString remoteName)
    {
        outbound.connectToServer(remoteName);
    }


signals:
    void pulseStartServer(QString serverName);
    void pulseSignal();
    void pulseReadSignal();
    void pulseReadSerialized();
    void pulseSendSerialized();
    void pulseShutdown();
    void pulseOutboundConnect(QString serverName);

public:
    QByteArray readArray;
    QByteArray writeArray;

    SignalType readSignal;
    QMutex receivedLock;
    QWaitCondition receivedWait;

    QMutex outboundLock;
    QWaitCondition outboundWait;

    QString key;
    QString serverName;
    QString remoteName;

    QLocalSocket *inbound;
    QLocalSocket outbound;
    QLocalServer server;


    void waitForInbound()
    {
        QMutexLocker locker(&receivedLock);
        while (!inbound || inbound->state() != QLocalSocket::ConnectedState) {
            bool res = receivedWait.wait(&receivedLock,30*1000);
            if (!res)
                qDebug() << key << " " << QThread::currentThread() << " waiting timed out, server thread is " << server.thread() << " base thread " << basis;
        }
    }

    void connectToRemote(const QString &remoteName)
    {
        emit pulseOutboundConnect(remoteName);

        QMutexLocker locker(&outboundLock);
        while (outbound.state() != QLocalSocket::ConnectedState) {
            outboundWait.wait(&outboundLock,30*1000);

        }
    }

    SignalType getSignal()
    {
        emit pulseReadSignal();
        return readSignal;
    }


    template<typename T>
    bool readData(T &input)
    {
        emit pulseReadSerialized();
        QDataStream deserializer(readArray);
        deserializer >> input;
        return true;
    }

    Transform *readTForm()
    {
        emit pulseReadSerialized();

        QByteArray data = readArray;
        QDataStream deserializer(data);
        Transform *res = Transform::deserialize(deserializer);
        sendSignal(CommunicationManager::OUTPUT_AVAILABLE);
        return res;
    }

    template<typename T>
    bool sendData(const T &output)
    {
        QBuffer buffer;
        buffer.open(QBuffer::ReadWrite);

        QDataStream serializer(&buffer);
        serializer << output;
        writeArray = buffer.data();
        emit pulseSendSerialized();
        return true;
    }

    SignalType sendType;
    void sendSignal(SignalType signal)
    {
        sendType = signal;
        if (QThread::currentThread() == this->thread() )
            this->sendSignalInternal();
        else
            emit pulseSignal();
    }

    void startServer(QString server)
    {
        emit pulseStartServer(server);
    }

    void shutdown()
    {
        emit pulseShutdown();
    }

 
};

class EnrollmentWorker : public QObject
{
    Q_OBJECT
public:
    CommunicationManager *comm;
    QString name;

    ~EnrollmentWorker()
    {
        delete transform;

        comm->shutDownThread();
        delete comm;
    }

    br::Transform *transform;

public:
    void connections(const QString &baseName)
    {
        comm = new CommunicationManager();
        name = baseName;
        comm->key = "worker_"+baseName.mid(1,5);
        comm->startServer(baseName+"_worker");
        comm->connectToRemote(baseName+"_master");

        comm->waitForInbound();

        transform = comm->readTForm();
    }

    void workerLoop()
    {
        QString sign = "worker " + name;
        CommunicationManager::SignalType signal;
        forever {
            signal= comm->getSignal();

            if (signal == CommunicationManager::SHOULD_END) {
                break;
            }
            TemplateList inList;
            TemplateList outList;

            comm->readData(inList);
            transform->projectUpdate(inList,outList);
            comm->sendData(outList);
        }
        comm->shutdown();
    }
};

void WorkerProcess::mainLoop()
{
    processInterface = new EnrollmentWorker();
    processInterface->transform = NULL;
    processInterface->connections(baseName);
    processInterface->workerLoop();
    delete processInterface;
}

class ProcessInterface : public QObject
{
    Q_OBJECT
public:
    QThread *basis;

    ~ProcessInterface()
    {
        basis->quit();
        basis->wait();
        delete basis;
    }

    ProcessInterface()
    {
        basis = new QThread();

        moveToThread(basis);
        workerProcess.moveToThread(basis);

        connect(this, SIGNAL(pulseEnd()), this, SLOT(endProcessInternal()), Qt::BlockingQueuedConnection);
        connect(this, SIGNAL(pulseStart(QStringList)), this, SLOT(startProcessInternal(QStringList)), Qt::BlockingQueuedConnection);

        basis->start();
    }

    QProcess workerProcess;
    void endProcess()
    {
        emit pulseEnd();
    }

    void startProcess(QStringList arguments)
    {
        emit pulseStart(arguments);
    }

signals:
    void pulseEnd();
    void pulseStart(QStringList);

protected slots:
    void endProcessInternal()
    {
        workerProcess.waitForFinished(-1);
    }

    void startProcessInternal(QStringList arguments)
    {
        workerProcess.setProcessChannelMode(QProcess::ForwardedChannels);
        workerProcess.start("br", arguments);
        workerProcess.waitForStarted(-1);
    }
};

struct ProcessData
{
    CommunicationManager comm;
    ProcessInterface proc;
    bool initialized;
    ProcessData()
    {
        initialized = false;
    }

    ~ProcessData()
    {
        if (initialized) {
            comm.sendSignal(CommunicationManager::SHOULD_END);
            proc.endProcess();
            comm.shutdown();
            comm.shutDownThread();
        }
    }
};


/*!
 * \ingroup transforms
 * \brief Interface to a separate process
 * \author Charles Otto \cite caotto
 */
class ProcessWrapperTransform : public WrapperTransform
{
    Q_OBJECT
    Q_PROPERTY(int concurrentCount READ get_concurrentCount WRITE set_concurrentCount RESET reset_concurrentCount STORED false)
    BR_PROPERTY(int, concurrentCount, 2)

    QString baseKey;

    Resource<ProcessData> processes;

    Transform *smartCopy(bool &newTransform)
    {
        newTransform = false;
        return this;
    }

    // make clang shut up? what a great compiler
    using WrapperTransform::project;
    using WrapperTransform::train;

    void project(const TemplateList &src, TemplateList &dst) const
    {
        if (src.empty())
            return;
        
        ProcessData *data = processes.acquire();
        if (!data->initialized)
            activateProcess(data);

        CommunicationManager *localComm = &(data->comm);

        localComm->sendSignal(CommunicationManager::INPUT_AVAILABLE);
        localComm->sendData(src);

        localComm->readData(dst);
        processes.release(data);
    }

    void train(const TemplateList &data)
    {
        (void) data;
    }

    void init()
    {
        processActive = false;
        serialized.clear();
        if (transform) {
            QDataStream out(&serialized, QFile::WriteOnly);
            transform->serialize(out);
            tcount = Globals->parallelism;
            counter.acquire(counter.available());
            counter.release(this->concurrentCount);
        }
    }

    static QSemaphore counter;
    mutable int tcount;
    mutable QByteArray serialized;
    void transmitTForm(CommunicationManager *localComm) const
    {
        if (serialized.isEmpty() )
            qFatal("Trying to transmit empty transform!");

        counter.acquire(1);
        static QMutex transmission;
        QMutexLocker lock(&transmission);
        tcount--;

        localComm->writeArray = serialized;
        if (tcount == 0)
            serialized.clear();
        lock.unlock();

        emit localComm->pulseSendSerialized();
        localComm->getSignal();
        counter.release(1);
    }

    void activateProcess(ProcessData *data) const
    {
        data->initialized = true;
        // generate a uuid for our local servers
        QUuid id = QUuid::createUuid();
        QString baseKey = id.toString();

        QStringList argumentList;
        // We serialize and transmit the transform directly, so algorithm doesn't matter.
        argumentList.append("-quiet");
        argumentList.append("-algorithm");
        argumentList.append("Identity");
        if (!Globals->path.isEmpty()) {
            argumentList.append("-path");
            argumentList.append(Globals->path);
        }
        argumentList.append("-parallelism");
        argumentList.append(QString::number(0));
        argumentList.append("-slave");
        argumentList.append(baseKey);

        data->comm.key = "master_"+baseKey.mid(1,5);

        data->comm.startServer(baseKey+"_master");

        data->proc.startProcess(argumentList);
        data->comm.waitForInbound();
        data->comm.connectToRemote(baseKey+"_worker");

        transmitTForm(&(data->comm));
    }

    bool timeVarying() const
    {
        return false;
    }

public:
    bool processActive;
    ProcessWrapperTransform() : WrapperTransform(false) { processActive = false; }
};
QSemaphore ProcessWrapperTransform::counter;

BR_REGISTER(Transform, ProcessWrapperTransform)

}

#include "core/processwrapper.moc"
